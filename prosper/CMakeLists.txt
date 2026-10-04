cmake_minimum_required(VERSION 3.20)
project(prosper LANGUAGES C CXX)   # C for the vendored LibAtrac9 decoder (third_party/libatrac9)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
include(cmake/WindowsGnuVariadicProducer.cmake)

# #4078: the native WinLibs POSIX emutls runtime can free C++ TLS storage before its
# destructor. This explicitly selected profile uses Clang's native Windows TLS instead.
# Keep ordinary builds unchanged; the lifetime regression remains visible on broken runtimes.
if(PROSPER_NATIVE_CLANG_TLS)
  if(NOT WIN32 OR NOT MINGW OR NOT CMAKE_C_COMPILER_ID STREQUAL "Clang" OR
     NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR
     NOT CMAKE_CXX_COMPILER_TARGET STREQUAL "x86_64-w64-windows-gnu")
    message(FATAL_ERROR "native-Clang TLS profile requires actual Windows GNU-ABI Clang compilers")
  endif()
  add_compile_options(-fno-emulated-tls)
  # libstdc++/libgcc from the GNU sysroot use winpthreads. Append after the compiler's
  # default runtime archives, rather than relying on archive order in Threads::Threads.
  set(CMAKE_CXX_STANDARD_LIBRARIES "${CMAKE_CXX_STANDARD_LIBRARIES} -lwinpthread")
  message(STATUS "prosper native-Clang profile: Windows GNU ABI, native TLS (not emutls)")
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Agentic-first: binaries must run with no PATH/DLL setup so CI (ctest) is hermetic.
# On mingw, statically link the GCC/C++ runtimes.
if(MINGW OR (WIN32 AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU"))
  add_link_options(-static -static-libgcc -static-libstdc++)
endif()

# --- Diagnostics (optional observer-only instrumentation) -----------------
# Boot timeline capture, event bus, JSON export. OFF by default: zero cost when
# disabled (stubs compile to no-ops). ON adds ~300 lines of optional code.
option(PROSPER_DIAGNOSTICS "Enable diagnostics instrumentation (boot timeline, event bus)" OFF)

# --- macOS Vulkan (MoltenVK) override ------------------------------------
# prosper on macOS is built x86_64 (to run the guest under Rosetta), but Homebrew's MoltenVK/loader
# are arm64-only, so find_package(Vulkan) would hand back an unlinkable arm64 lib. Point the build at
# a UNIVERSAL (x86_64+arm64) MoltenVK instead — either the LunarG Vulkan SDK, or the prebuilt
# KhronosGroup/MoltenVK release (`MoltenVK-macos.tar`, universal). Pass:
#   -DPROSPER_MACOS_MOLTENVK=/path/to/libMoltenVK.dylib   (universal)
#   -DPROSPER_MACOS_VULKAN_INCLUDE=/path/to/vulkan/headers (dir containing vulkan/vulkan.h)
# When set we pre-create the Vulkan::Vulkan imported target from it, so every later
# find_package(Vulkan) is skipped (guarded on `NOT TARGET Vulkan::Vulkan`). Headers are
# architecture-independent, so Homebrew's vulkan-headers work for the include dir.
if(APPLE AND PROSPER_MACOS_MOLTENVK)
  if(NOT PROSPER_MACOS_VULKAN_INCLUDE)
    set(PROSPER_MACOS_VULKAN_INCLUDE "/opt/homebrew/include")   # Homebrew vulkan-headers (arch-independent)
  endif()
  add_library(Vulkan::Vulkan UNKNOWN IMPORTED)
  set_target_properties(Vulkan::Vulkan PROPERTIES
    IMPORTED_LOCATION "${PROSPER_MACOS_MOLTENVK}"
    INTERFACE_INCLUDE_DIRECTORIES "${PROSPER_MACOS_VULKAN_INCLUDE}")
  set(Vulkan_FOUND TRUE)
  set(Vulkan_LIBRARY "${PROSPER_MACOS_MOLTENVK}")
  set(Vulkan_INCLUDE_DIR "${PROSPER_MACOS_VULKAN_INCLUDE}")
  # The app dlopen's / links MoltenVK by leaf name; add its dir to the executables' rpath so
  # SDL_Vulkan_LoadLibrary("libMoltenVK.dylib") and the direct link both resolve at runtime.
  get_filename_component(_mvk_dir "${PROSPER_MACOS_MOLTENVK}" DIRECTORY)
  set(CMAKE_BUILD_RPATH "${_mvk_dir}")
  message(STATUS "macOS Vulkan: using universal MoltenVK ${PROSPER_MACOS_MOLTENVK}")
endif()

# --- tools ---------------------------------------------------------------
# self_dump: SELF/ELF inspector and NID import extractor (M0).
add_executable(self_dump tools/self_dump/self_dump.cpp)

# index_fetch_probe: bare-Vulkan oracle for indexed-draw index fetch (#2961). The #2961 hypothesis
# (RADV/STRIX_HALO reading sequential ordinals) was falsified with this tool; it stays so any
# future suspicion costs one command. Standalone on purpose — no prosper code, so it cannot
if(NOT TARGET Vulkan::Vulkan)
  find_package(Vulkan QUIET)
endif()
if(TARGET Vulkan::Vulkan)
  add_executable(index_fetch_probe tools/index_fetch_probe/probe.c)
  target_link_libraries(index_fetch_probe PRIVATE Vulkan::Vulkan)
  # pixel_history_control: the known-answer construction that pixel_history.py is validated
  # against (see tools/pixel_history/AGENTS.md). Built here without the capture bracket so
  # its SELF-check -- "did my five arms render as designed?" -- runs with no RenderDoc at
  # all. The capture build is opt-in in tools/doctor/CMakeLists.txt.
  add_executable(pixel_history_control tools/pixel_history/control.c)
  target_link_libraries(pixel_history_control PRIVATE Vulkan::Vulkan)
endif()

# --- third-party: LibAtrac9 (MIT, vendored) - ATRAC9 audio decode -------------------------
# Vendored verbatim (see third_party/libatrac9/README.md). C99; needs libm. Compiled as its own
# static lib so the C sources stay isolated from prosper's C++ and are trivially auditable/updatable.
file(GLOB LIBATRAC9_SRC CONFIGURE_DEPENDS third_party/libatrac9/src/*.c)
add_library(libatrac9 STATIC ${LIBATRAC9_SRC})
target_include_directories(libatrac9 PUBLIC third_party/libatrac9/src)
set_target_properties(libatrac9 PROPERTIES C_STANDARD 99)
if(NOT WIN32)
  target_link_libraries(libatrac9 PUBLIC m)
endif()

# --- core library (M1+) --------------------------------------------------
# Loader / HLE / kernel / gpu sources accrete here per the roadmap.
include(cmake/ProsperBuildRevision.cmake)
prosper_add_build_revision_library(prosper_build_revision WORK_TREE "${CMAKE_SOURCE_DIR}/..")

file(GLOB_RECURSE PROSPER_SRC CONFIGURE_DEPENDS src/*.cpp)
if(WIN32)
  list(FILTER PROSPER_SRC EXCLUDE REGEX "src/host/memory/committed_section_unavailable\\.cpp$")
else()
  list(FILTER PROSPER_SRC EXCLUDE REGEX "src/host/memory/committed_section_win\\.cpp$")
endif()
if(PROSPER_NATIVE_CLANG_TLS)
  list(FILTER PROSPER_SRC EXCLUDE REGEX "src/hle/libc/libc_variadic_capture\\.cpp$")
endif()
# Exclude the diagnostics HISTORY/REPORT layer when the feature is disabled (default). When enabled,
# the PROSPER_DIAGNOSTICS define activates the real implementation; the headers always compile
# (stubs when disabled).
#
# Scoped to core/ and storage/ deliberately: src/diagnostics/boot_phase_log.cpp must be in EVERY
# build. It is the one file here that answers "where did boot_program() stop?", and boot_program()
# runs guest code (run_guest_inits), so a title can sit inside it indefinitely while every frontend
# prints nothing. An instrument for that has to exist in the default build — excluding it with the
# rest of the folder is how the boot phases came to be unprintable in every build the project
# shipped. See src/diagnostics/AGENTS.md.
if(NOT PROSPER_DIAGNOSTICS)
  list(FILTER PROSPER_SRC EXCLUDE REGEX "src/diagnostics/(core|storage)/")
endif()
if(PROSPER_SRC)
  add_library(prosper_core STATIC ${PROSPER_SRC})
  target_include_directories(prosper_core PUBLIC src)
  # stb_truetype.h backs the libSceFont Ft (FreeType-edition) rasterizer in
  # src/hle/util/hle_font.cpp. PRIVATE: it is an implementation detail of one translation unit, not
  # part of prosper_core's interface. See third_party/stb/README.md -- unlike stb_image.h next to
  # it, THIS header is on the guest path.
  target_include_directories(prosper_core PRIVATE third_party/stb)
  target_link_libraries(prosper_core PUBLIC libatrac9 PRIVATE prosper_build_revision)
  target_compile_definitions(prosper_core PRIVATE PROSPER_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
  if(PROSPER_NATIVE_CLANG_TLS)
    prosper_add_gnu_variadic_producer(prosper_core
      "${CMAKE_CURRENT_SOURCE_DIR}/src/hle/libc/libc_variadic_capture.cpp" libc_capture)
  endif()
  if(PROSPER_DIAGNOSTICS)
    target_compile_definitions(prosper_core PRIVATE PROSPER_DIAGNOSTICS)
  endif()
  find_package(Threads REQUIRED)
  target_link_libraries(prosper_core PUBLIC Threads::Threads)
  # Windows: the guest futex HLE (sceKernelWaitOnAddress/WakeByAddress) is backed by the native Win32
  # WaitOnAddress/WakeByAddressSingle/All, which live in the Synchronization import lib (not kernel32).
  if(WIN32)
    target_link_libraries(prosper_core PUBLIC synchronization bcrypt)
  endif()
endif()

# Bounded interactive performance capture (F8): pure host-side state + serialization shared by the
# SDL app and the live graphics/compute backends. It deliberately has no SDL/Vulkan dependency, so
# its ring, bounds, and atomic file contract run in ordinary headless CI.
add_library(prosper_performance_capture STATIC frontends/shared/perf/performance_capture.cpp)
target_include_directories(prosper_performance_capture PUBLIC frontends src)
target_link_libraries(prosper_performance_capture PRIVATE prosper_build_revision)
# Windows: the process memory sample (working set + commit charge) calls GetProcessMemoryInfo, which
# lives in the PSAPI import library rather than the kernel32 set every target links by default.
if(WIN32)
  target_link_libraries(prosper_performance_capture PRIVATE psapi)
endif()

# --- tests (agentic-first: self-checking, exit-code = truth) -------------
enable_testing()
include(cmake/ProsperGTest.cmake)

# Pure selection/accounting policy behind the stable compute-timing hash filter. This must remain
# runnable without prosper_core, a game dump, or Vulkan so a broken diagnostic cannot hide behind a
# skipped execution test on a host without graphics support.
add_executable(test_storage_write_mask_spirv frontends/shared/tests/test_storage_write_mask_spirv.cpp)
target_include_directories(test_storage_write_mask_spirv PRIVATE frontends)
add_test(NAME storage_write_mask_spirv COMMAND test_storage_write_mask_spirv)

add_executable(test_compute_timing_selector tests/shared/compute/test_compute_timing_selector.cpp)
target_include_directories(test_compute_timing_selector PRIVATE frontends)
add_test(NAME compute_timing_selector_policy COMMAND test_compute_timing_selector)

# execute_item()'s five CPU phases on a dispatch that breaks early (#3461): no negative interval,
# no whole-item cleanup, and the parts still sum to the item.
prosper_add_gtest(test_compute_phase_attribution
  SOURCES tests/shared/compute/test_compute_phase_attribution.cpp
  INCLUDES frontends)

# Pure fail-closed policy for a behavior-neutral census of hypothetical deferred compute results.
# It has no Vulkan dependency and does not change the live backend's synchronous writeback.
add_executable(test_compute_authority_census tests/shared/compute/test_compute_authority_census.cpp)
target_include_directories(test_compute_authority_census PRIVATE frontends)
add_test(NAME compute_authority_census_policy COMMAND test_compute_authority_census)

# The exactness proof behind the differential buffer upload (#3696): the upload copies only the
# extent compute_buffers_diff_span reports, so a span one byte too narrow leaves a stale byte in a
# GPU buffer. Pure CPU and header-only, so it registers here and runs on every platform -- it used to
# link prosper_live_renderer and was therefore Linux-with-Vulkan only (#3699).
#
# Do NOT read that as race coverage for the eight std::thread workers and their atomic combine:
# `Host TSan` in .github/workflows/ci.yml configures `-S prosper/tools/doctor`, a separate CMake
# project that does not build prosper, so this code is not under TSan.
# compute_buffer_bytes.hpp counts parallel-worker spawn volume, so it reaches into
# src/diagnostics; any target including it needs `src` and that translation unit.
find_package(Threads REQUIRED)
prosper_add_gtest(test_compute_buffer_diff_span
  SOURCES tests/shared/compute/test_compute_buffer_diff_span.cpp
          src/diagnostics/worker_spawn_census.cpp
          src/diagnostics/exit_census.cpp
          src/diagnostics/transfer_pressure.cpp
          src/diagnostics/exit_reports.cpp
  INCLUDES frontends src
  LIBRARIES Threads::Threads)

# Standard-library-only host lifetime guard: no core/GPU/game and no scratch/export opt-out.
add_executable(test_native_tls_lifetime tests/host/platform/test_native_tls_lifetime.cpp)
target_link_libraries(test_native_tls_lifetime PRIVATE Threads::Threads)
add_test(NAME native_tls_lifetime COMMAND test_native_tls_lifetime)
set_tests_properties(native_tls_lifetime PROPERTIES TIMEOUT 30)

find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND)
  add_test(NAME diagnostic_doctor
           COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/doctor/test_doctor.py")
  # The verdict logic only. The replay half needs RenderDoc and a GPU and cannot run here,
  # which is exactly why the shader-output suppression was lifted into a pure function that
  # this can reach -- see the comment on suppress_shader_output().
  add_test(NAME pixel_history
           COMMAND "${Python3_EXECUTABLE}"
                   "${CMAKE_SOURCE_DIR}/tools/pixel_history/test_pixel_history.py")
  # self_dump --import-slots reads DT_JMPREL/DT_RELA to answer "which GOT slot does this Sony import
  # land in" (#1625). The fixtures are PS5-shaped ELFs written byte by byte in the test, so they are
  # an independent statement of the relocation layout rather than anything self_dump produced, and
  # they run in CI with no game dump. Includes the zero-result arms: a tool that answers "nothing"
  # when it means "I did not parse anything" is the defect class of #2399.
  add_test(NAME self_dump_import_slots
           COMMAND "${Python3_EXECUTABLE}"
                   "${CMAKE_SOURCE_DIR}/tools/self_dump/test_import_slots.py"
                   "$<TARGET_FILE:self_dump>")
  add_test(NAME re_xref_decoder
           COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/re/test_xref.py")
  # The GPU memory budget's answer is only as complete as its wiring, and nothing in the compiler
  # enforces that wiring: a new vkAllocateMemory beside an existing one compiles, links, runs, and is
  # simply absent from the count. An unwired FREE is the worse half -- those bytes are never given
  # back, so the figure climbs and reads as a leak that is not there (#3533).
  add_test(NAME vkmem_coverage
           COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/vkmem_coverage.py")
  # survey_sizes.py's classifier decides which refactoring tool can act on a file, and it has been
  # wrong five times: on the lambda boundary, by calling a file of many small functions "nothing to
  # do", by measuring the lambda share over the enclosing namespace, by inferring unparsed lines
  # from region kinds (which cannot see a mid-file #if), and by not mirroring regions_of's descent
  # rule. The last three were in how the classifier's
  # INPUTS are measured, so no arithmetic case could have caught them; they are pinned by fixtures
  # under tools/refactor/testdata/ that the selftest scores ONLY where libclang is importable, and
  # it says loudly when it skips them. libclang resolves lazily, so the arithmetic half runs here
  # even though the surveying itself needs a libclang the runner does not have.
  # map_symbols' clustering decides where a split cuts, and its failure mode is a confident wrong
  # proposal rather than a crash -- one draft collapsed a 6,515-line file into a single 6,374-line
  # "part" and reported a 0% cut. Pure arithmetic over the reference matrix, and libclang resolves
  # lazily there now, so this runs on a runner that has none.
  # split_file.py was in NO ctest case at all, so a "3/3 refactor tests pass" said nothing about
  # the splitter -- including a cross-reference check whose arms could not fail. Both gaps found in
  # the same review.
  add_test(NAME refactor_split_file
           COMMAND "${Python3_EXECUTABLE}"
                   "${CMAKE_SOURCE_DIR}/tools/refactor/split_file.py" --selftest)
  add_test(NAME refactor_map_symbols_clusters
           COMMAND "${Python3_EXECUTABLE}"
                   "${CMAKE_SOURCE_DIR}/tools/refactor/map_symbols.py" --selftest)
  add_test(NAME refactor_survey_classifier
           COMMAND "${Python3_EXECUTABLE}"
                   "${CMAKE_SOURCE_DIR}/tools/refactor/survey_sizes.py" --selftest)
  # census_bodies ranks FUNCTION and CLASS bodies, which is the measurement survey_sizes cannot
  # make: survey_sizes names a file's dominant region and stops, so it cannot rank two files' worst
  # functions against each other. Its arithmetic half (line spans, the lambda-share union, the
  # verdict thresholds) runs with no libclang; the parse arm is skipped there rather than folded in.
  add_test(NAME refactor_census_bodies
           COMMAND "${Python3_EXECUTABLE}"
                   "${CMAKE_SOURCE_DIR}/tools/refactor/census_bodies.py" --selftest)
  # outline_methods moves a class's method bodies out of a header. Its arithmetic half -- where the
  # declarator is, which methods are refusable, and that a moved body is carried verbatim -- is pure
  # text and runs with no libclang, which is what lets it be a registered case rather than something
  # somebody remembers to run.
  # The header this guards is compiled by eleven translation units, so a method written inline in
  # the class body costs eleven parses and is invisible in review. Textual and dependency-free on
  # purpose: it must fail for whoever adds the next one, including with no libclang present.
  add_test(NAME spirv_compute_header_shape
           COMMAND "${Python3_EXECUTABLE}"
                   "${CMAKE_SOURCE_DIR}/tests/gpu/recompiler/test_spirv_compute_header_shape.py")
  add_test(NAME refactor_outline_methods
           COMMAND "${Python3_EXECUTABLE}"
                   "${CMAKE_SOURCE_DIR}/tools/refactor/outline_methods.py" --selftest)
  # The measurement fixtures are a SEPARATE case so that a runner without libclang reports them as
  # Skipped rather than folding them into a green Passed. A pass whose domain is smaller than the
  # local one, with nothing in the report saying so, is the hazard this project records for its GPU
  # tests under CI's software rasteriser.
  add_test(NAME refactor_survey_measurement
           COMMAND "${Python3_EXECUTABLE}"
                   "${CMAKE_SOURCE_DIR}/tools/refactor/survey_sizes.py" --selftest-parse)
  set_tests_properties(refactor_survey_measurement PROPERTIES SKIP_RETURN_CODE 77)
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    add_test(NAME profiling_access
             COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/profiling_access/test_access.py")
    set_tests_properties(profiling_access PROPERTIES TIMEOUT 30)
  endif()
  # The human-authored snapshot store. Runs with no dump and no emulator: every arm builds its own
  # BMPs. The arms are chosen around SILENT failure -- a signature that cannot separate two scenes
  # would pass every regression, and an accept() that moved the signature without the reference
  # image would show a misleading "expected" at exactly the moment someone is deciding whether to
  # trust it.
  add_test(NAME snapshot_snaps_tool
           COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/snapshot/test_snaps.py")
  # iostore_index turns a .ucas read offset into a package name, which is the only instrument that
  # separates "reached gameplay" from "sat on an animating menu" on an Unreal title. Its failure
  # mode is a plausible WRONG name rather than an error, so the self-test pins the two-step
  # physical->logical->chunk mapping on a fixture where the two offset spaces genuinely differ --
  # a naive "physical == logical" reading passes every other check and misnames most of a stream.
  add_test(NAME re_iostore_index
           COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/re/iostore_index.py"
                   "--self-test")
  # worktree_reclaim deletes directories, so its tests are mutation arms first: each builds a tree
  # that MUST NOT be removed (dirty, untracked-only, unmerged, held by a live process, recently
  # touched, locked, nested) and asserts both the refusal and which guard produced it. It also
  # pins the instrument bug found while writing it -- examining a worktree used to make it look
  # freshly touched, which reported all 278 real trees as busy (#1735).
  #
  # Linux only, and that is a statement about the tool rather than about the test. Its in-use
  # guard reads /proc, so on macOS and Windows it cannot prove nobody is inside a tree and fails
  # closed: it classifies, and removes nothing, there. Testing removal guards where removal never
  # happens would assert on a configuration nobody runs. The fail-closed behaviour itself is
  # covered by a platform-independent arm, so the property that protects those platforms is still
  # verified everywhere this suite runs.
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    add_test(NAME worktree_reclaim_tool
             COMMAND "${Python3_EXECUTABLE}"
                     "${CMAKE_SOURCE_DIR}/tools/test_worktree_reclaim.py")
  endif()

  # lanekill SIGNALS processes, so its arms are refusals first: a process in another worktree, a
  # path that attributes to nothing, a lookalike directory sharing a name prefix, and an
  # end-to-end run where exactly one of two live processes dies.
  #
  # That end-to-end arm exists because the first version of this suite tested only ATTRIBUTION.
  # A reviewer mutated the tool six ways -- each turning it into a cross-lane killer -- and every
  # arm stayed green, because none of them reached the code that decides to send a signal. Each
  # of those mutations now reddens, verified by re-running them.
  #
  # Linux-only, and that is a statement about the tool: attribution reads /proc. The fail-closed
  # arm is Linux-only TOO, which is a deliberate narrowing -- it used to run everywhere by
  # installing a stub and asserting the stub's own return value, which said nothing about
  # lanekill and stayed green when the gate was deleted. It now runs main() against a live
  # process in its own worktree and asserts the refusal spared it, which needs a real process.
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    add_test(NAME lanekill_tool
             COMMAND "${Python3_EXECUTABLE}"
                     "${CMAKE_SOURCE_DIR}/tools/test_lanekill.py")
  endif()
  # wt_stash parks uncommitted work in refs/worktree/, which git documents as per-worktree, because
  # refs/stash is NOT: every worktree of this repo shares one stack, so two lanes each obeying the
  # "work in your own worktree" rule still pop each other's entries (#3174, instrument trap 247).
  #
  # The first arm is a positive control built by hand from git primitives: two linked worktrees,
  # plain `git stash` in each, and an assertion that lane A's pop really does hand it lane B's
  # content. Every later arm claims the tool removes that failure, and none of them means anything
  # if the fixture cannot produce it -- so a failure there reports VOID rather than a red arm.
  #
  # Platform-independent deliberately. Unlike worktree_reclaim and lanekill above, nothing here
  # reads /proc: the whole subject is ref namespacing, which behaves the same everywhere git runs,
  # and the collision this pins is likeliest on the Windows/PowerShell route the charter also
  # prescribes.
  add_test(NAME wt_stash_tool
           COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/test_wt_stash.py")
  add_test(NAME re_edis_mapping
           COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/re/test_edis.py")
  # nid_gate_scan answers "does this title BRANCH on an imported function's return value?", and a
  # wrong answer there is a plausible-looking table rather than a visible failure. Three of its
  # cases are regressions for classifier bugs found only after a table had been published (#2002).
  add_test(NAME re_nid_gate_classifier
           COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/re/test_nid_gate_scan.py")
  # stub_nid_map inverts that resolution to answer "which import is the stub at address X?" while
  # reading a call chain. Both ways it can be wrong are silent: a dropped stub entry point makes a
  # real import report NOT A STUB, and a dropped fail-visible line makes "not an import" and "wrong
  # address" the same empty output (#2186).
  add_test(NAME re_stub_nid_map
           COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/re/test_stub_nid_map.py")
  # hle_handler_map answers "how many Sony entry points collapse onto ONE prosper handler?" — the
  # blind spot a runtime return-value histogram cannot see, because it keys on the handler. Every
  # way the parser can be wrong yields a plausible table rather than a failure, so the shapes, the
  # `#if` arms and the per-name (not per-site) counting are all pinned (#2070).
  add_test(NAME re_hle_handler_map
           COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/re/test_hle_handler_map.py")
  # ...and reconciled against the registry the compiled binary actually builds, so a registration
  # SHAPE the parser does not recognise fails loudly instead of quietly shrinking the census.
  add_test(NAME re_hle_handler_map_reconcile
           COMMAND "${Python3_EXECUTABLE}"
                   "${CMAKE_SOURCE_DIR}/tools/re/test_hle_registry_reconcile.py"
                   "$<TARGET_FILE:hle_registry_dump>")
  # prx_to_elf.py writes an ELF header by hand, four bytes apart from a field that means something
  # else entirely — #2016 put the program-header count in the section-header-count slot, and the
  # result was a malformed image no consumer complained about until binutils refused the file. The
  # test reads the produced bytes at their gABI offsets; nothing else can see this class of bug.
  add_test(NAME il2cpp_prx_to_elf_header
           COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/il2cpp/test_prx_to_elf.py")
  # resolve.py's other half: the nearest-preceding lookup that turns an address into a method NAME.
  # Every way it can be wrong is silent -- a window off by one, the bisect tie rule, or a dropped
  # record filter each yield a plausible name rather than an error, and that name is what a reader
  # trusts to place a breakpoint. The window arms sit exactly ON the 0x8000 edge for that reason:
  # an arm a byte outside it passes whether the comparison is `<` or `<=`, which is how the first
  # version of this file reported a green boundary test it could not actually fail (#2526).
  add_test(NAME il2cpp_resolve_lookup
           COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/il2cpp/test_il2cpp_tools.py")
  # replay_determinism_report decides whether a renderer-determinism campaign may be read as a
  # NEGATIVE. #2945's failure rate drifts machine-wide over minutes, so a campaign that sits in a
  # quiet window and a campaign on a repaired renderer produce identical numbers -- which is why
  # the tool has three verdicts and the third (UNDECIDED: one hash, but the bare-Vulkan control
  # never failed either) is the one being pinned. Every arm here except one still passes with the
  # control check deleted, so that arm is a MUTATION: it defeats the check and requires the quiet
  # campaign to flip to DETERMINISTIC.
  #
  # Platform-independent, and that is a statement about the tool: this half touches no Vulkan, no
  # GPU and no capture. It asserts on the READING of a campaign, which is where both of this
  # issue's published misreadings happened.
  add_test(NAME replay_determinism_report
           COMMAND "${Python3_EXECUTABLE}"
                   "${CMAKE_SOURCE_DIR}/tools/determinism/test_replay_determinism_report.py")
  # hle_calls decides, per run, whether a reader should TRUST an entire measurement -- a nine-state
  # verdict machine whose most consequential branch is VOID vs VIOLATED ("this run cannot say" vs
  # "distrust every value in it"). It had no test at all, because hle_calls_gdb.py does `import gdb`
  # at module scope and nothing in it was reachable outside a gdb process. #2053 lifted the verdict
  # into hle_calls_control.py, which the gdb script now forwards to, so this asserts on the code that
  # actually runs rather than on a copy.
  add_test(NAME hle_calls_positive_control
           COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/hle_calls/test_hle_calls.py")
  # The sibling above is pure; this one cannot be. #2075 was not a logic bug: `--values` recorded
  # NOTHING on prosper's own default build, because gdb's FinishBreakpoint.return_value comes from
  # the DWARF return type and `Release` carries no `-g`. Only a real gdb over a real no-debug-info
  # binary reproduces that, so this compiles test_values_fixture.cpp twice (-g0 and -g) and runs the
  # real driver over each. A missing gdb/compiler exits 77 -> reported as `(Skipped)` by name, which
  # is the one thing a silently-skipped end-to-end test cannot do.
  add_test(NAME hle_calls_value_capture
           COMMAND "${Python3_EXECUTABLE}"
                   "${CMAKE_SOURCE_DIR}/tools/hle_calls/test_hle_calls_values.py")
  set_tests_properties(hle_calls_value_capture PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 600)
  add_test(NAME snapshot_guard_logic
           COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/snapshot/test_snapshot.py")
  add_test(NAME gpu_replay_regress_logic
           COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/gpu_replay/test_regress.py")
  # Half its cases are CORRECT shapes an earlier revision of the checker wrongly rejected. A doc
  # check that fires on valid input gets deleted rather than heeded, so the false-positive cases
  # are what keep it alive.
  add_test(NAME doc_table_checker
           COMMAND "${Python3_EXECUTABLE}"
                   "${CMAKE_SOURCE_DIR}/tools/docs/test_check_numbered_table.py")
  # Every `trap NNN` reference in the repository must name a row that exists. Half this suite is
  # FALSE-POSITIVE cases (`s_trap 1`, a bare number, "trapdoor"), because this checker scans every
  # tracked file and one that fires on ordinary prose or on an RDNA2 mnemonic gets disabled.
  add_test(NAME trap_citation_checker
           COMMAND "${Python3_EXECUTABLE}"
                   "${CMAKE_SOURCE_DIR}/tools/docs/test_check_trap_citations.py")
  # The trap-number allocator (#1729). Its network half is deliberately untested — a test needing
  # authentication gets disabled — but its LAST section pins its table parser against
  # check_numbered_table's, on the repository's real table. Those are two independent parsers, and a
  # divergence would hand out a number the gate then rejects, on somebody else's PR, hours later.
  add_test(NAME trap_number
           COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/docs/test_trap_number.py")
  # The PROGRESS_TRACKER.md generator's parser (#2730). Only the --selftest arm is registered, not
  # the live regeneration: reading the trackers needs `gh` and authentication, and a test that
  # needs credentials gets disabled -- CI runs the live --check itself, in the Docs job. What is
  # registered is the half that can fail SILENTLY: the parser accepts a compliant tracker and
  # rejects seven hand-written violations, so a regex that quietly widened (and would therefore
  # emit a plausible WRONG file rather than an error) fails here instead of shipping.
  add_test(NAME progress_tracker_generator
           COMMAND "${Python3_EXECUTABLE}"
                   "${CMAKE_SOURCE_DIR}/tools/docs/gen_progress_tracker.py" --selftest)
  # The pre-merge merge-result check (#2211). Its cases build real throwaway git repositories and
  # run real merges: the git plumbing IS the tool, so a mocked merge would test nothing. The case
  # that earns its keep is the clean merge whose RESULT holds a duplicate -- the state no head-time
  # check can observe, because at head time the other branch's row did not exist.
  add_test(NAME doc_merge_result_checker
           COMMAND "${Python3_EXECUTABLE}"
                   "${CMAKE_SOURCE_DIR}/tools/docs/test_check_merge_result.py")
  # Parsing/gating logic of the Vulkan validation guard (#1704). Pure — it does not create a Vulkan
  # instance, so it runs everywhere. A parser that stops matching the layer's message framing would
  # report a clean suite forever, which is the one way this guard can fail silently.
  add_test(NAME vkval_scan_logic
           COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/vkval/test_vk_validation_scan.py")
  # ctest EXITS 0 when it finds no tests, so "nothing ran" and "everything passed" share a status
  # (#2187). Four CI jobs and both author-verification runs were reading that status. They now pass
  # --no-tests=error; this registers the checker that keeps the next one from omitting it. The
  # --selftest arm is what is registered rather than the repository scan, because a scan needs the
  # checkout root and this test must run from any build directory -- CI runs the scan itself, in
  # the Docs job. Its own failure mode is silence: a matcher that stops matching reports a clean
  # tree forever, which is why the selftest carries the prose lines an earlier revision got wrong.
  # New tests must use GoogleTest (prosper_add_gtest); hand-rolled main() files are allowlisted legacy.
  add_test(NAME gtest_policy
           COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/ci/check_gtest_policy.py"
                   --selftest)
  # New Python tests must use pytest conventions; legacy tests are allowlisted.
  add_test(NAME pytest_policy
           COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/ci/check_pytest_policy.py"
                   --selftest)
  add_test(NAME ctest_gate_checker
           COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/ci/check_ctest_gate.py"
                   --selftest)
  # `print(__doc__)` is a usage message only while the header is a docstring; rewritten as `#`
  # comments it prints the literal string `None` and the tool still exits non-zero, so nothing
  # looks broken and the caller is told nothing (#2399). Three tools have shipped that way, the
  # last of them twice -- resolve.py regressed in #2642 inside the very change that added a line
  # to its usage block, with nine checks green, because the only guard in the tree named xref.py.
  # So this asks the whole tree instead of one file. Unlike ctest_gate_checker above, the scan
  # itself is registered rather than only the selftest: the root it needs is ${CMAKE_SOURCE_DIR},
  # known at configure time, so it runs from any build directory. --selftest runs beside it
  # because a walk that matches nothing and a clean tree otherwise print the same thing.
  add_test(NAME tools_usage_text
           COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/ci/check_usage_text.py"
                   --selftest --root "${CMAKE_SOURCE_DIR}")
  # The merge gate (#3259). Its arms are all REFUSALS, and a gate that refuses everything would
  # satisfy every one of them -- so the suite leads with an all-pass positive control, and each
  # refusal asserts on the REASON rather than on the boolean. One arm reconstructs a merge that
  # actually went wrong: a red `Windows MinGW` read as green because a shell split the check NAME
  # on whitespace and landed on `MinGW` where the status belongs (#3234). The stale-head arm is
  # NOT a reconstruction -- #3259 first cited #3243 as one and that was wrong on the dates -- it
  # guards a state this repository produces routinely, since every lane pushes after CI starts.
  add_test(NAME pr_merge_gate
           COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/ci/test_pr_merge_gate.py")
  # check_python_quality.py ratchets ruff on CHANGED Python files only (#4126). Its tests skip,
  # loudly, when ruff is absent; the python-quality CI job installs it, so there they always run.
  add_test(NAME python_quality_gate
           COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/ci/test_check_python_quality.py")
  # check_cpp_lint.py gates clang-format/clang-tidy on CHANGED lines only (#4210). The end-to-end
  # arm skips, loudly, without the clang tools; the Linux CI job runs its --selftest directly.
  add_test(NAME cpp_lint_gate
           COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/ci/test_check_cpp_lint.py")
  # check_arch_ratchet.py: per-file counts of ten measured costs may go down, never up. The
  # tests build one positive instance of every rule by hand, so a matcher that quietly stops
  # matching goes red here instead of reporting a clean tree. The real-tree arms need the checkout
  # (`git ls-files`): full mode must EVALUATE, and delta mode against origin/main (skipped when
  # that ref is absent) must pass -- a stale row on main is not this build's failure.
  add_test(NAME arch_ratchet_gate
           COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/ci/test_check_arch_ratchet.py")
  # pr_body.py answers "is the PR body actually what I published?" (#2918). `gh pr edit --body-file`
  # was observed returning rc=1 on a GraphQL projects-deprecation error WITHOUT applying the edit,
  # leaving a stale body the author believed was corrected -- and a PR body is where reviewers and
  # the merge step take the author's account of the change.
  #
  # Every arm is hermetic: `pr_body.run` is replaced by a fake, so a write that reports success
  # while changing nothing is reachable on demand rather than by waiting for GitHub to misbehave.
  # That arm is the one that matters -- an implementation trusting the write's exit code passes
  # every other arm in the file. Platform-independent: nothing here reads /proc or shells out.
  add_test(NAME pr_body_readback
           COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/ci/test_pr_body.py")
  # ratchet_hook.py is the PreToolUse hook that runs check_arch_ratchet.py before an agent's
  # `git commit` / `git push` (#4193), in delta mode against origin/main. The rollout is warn-only
  # (BLOCK_ON_VIOLATION): a violation exits 0 with a systemMessage, and everything that is not a
  # verdict fails open. Each case -- clean, violation (warned, and blocked with the switch on),
  # could-not-evaluate, missing checker, no origin/main, not a checkout -- has an arm against a
  # throwaway git repo and a stand-in checker. The last arm reads .claude/settings.json: an
  # interpreter `||` chain there would swallow exit 2 once the switch is flipped.
  add_test(NAME ratchet_hook
           COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/ci/test_ratchet_hook.py")
  add_test(NAME session_start
           COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/test_session_start.py")
  # PROSPER_ENV_ON/_VALUE sample getenv once and return it forever. Applied to a name a TEST arms at
  # runtime, the arm becomes invisible -- and the test does not fail, it goes vacuous and keeps
  # printing [ok], which is why #2214 surfaced only as one unrelated-looking Vulkan binding error on
  # a single CI job. The scanner carries its own self-tests because its failure mode is silence: a
  # pattern that stops matching reports a clean tree forever. Three of those cases cover the wrapper
  # call forms (set_env/unset_env), which the first revision missed -- it passed a tree that still
  # contained the defect.
  add_test(NAME cached_env_arming_logic
           COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/env/check_cached_env.py"
                   "${CMAKE_SOURCE_DIR}")
  # The coverage half of test_env_numeric_sites (#3267). That test MIRRORS each converted site
  # rather than calling it -- the sites cache their reads in function-local statics, so a runtime
  # arm would go vacuous exactly as cached_env_arming_logic above describes -- and a mirror drifts
  # silently: convert a site, forget the arm, suite stays green. This refuses that by comparing the
  # set of knobs parsed through diagnostics/env_numeric.hpp against the set the test arms. It is a
  # text scan, so it needs nothing built.
  add_test(NAME env_numeric_arms
           COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/env/check_env_numeric_arms.py"
                   "${CMAKE_SOURCE_DIR}")
  # stdout carries BYTES, and Windows decodes them as cp1252/cp437. One em dash in a printf literal
  # therefore reaches a Windows user as three mojibake characters, so a grep for the line the tool
  # itself printed does not match it -- and a test asserting on the rendered string fails on
  # Windows only, which is how
  # #2579 shipped five of them past a green Linux CI (#2588). Also runs in the Docs job, which
  # needs nothing built: this is a text scan, and a defect it can see should not wait for a build.
  # The scanner self-tests its own parsing AND its line numbers, because both failure modes are
  # silent -- a pattern that stops matching reports a clean tree forever, and a drifted line counter
  # sends the reader to innocent code while the real site goes unexamined (the first revision did).
  add_test(NAME ascii_output_literals
           COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/output/check_ascii_output.py"
                   "${CMAKE_SOURCE_DIR}")
  # A diagnostic whose PRODUCER is armed by a different switch than its PRINTER reports unarmed
  # state as a confident, well-formed ZERO -- indistinguishable from a measured negative, and
  # exactly what the interesting hypothesis usually predicts (#2149; #2111 emitted 183 such
  # negatives). Two arms: --selftest pins the three signatures against reduced forms of the four
  # MEASURED instances and runs from any build directory; the repository scan compares the tree
  # against tools/env/diag_gate_baseline.txt and runs in the Docs CI job, because it needs the
  # checkout root. Half the selftest cases assert on what must NOT match -- a checker that fires
  # on sound code gets regenerated past, not heeded -- and its own failure mode is silence: a
  # signature that stops matching reports a clean tree forever.
  add_test(NAME diag_gate_selftest
           COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/env/check_diag_gates.py"
                   --selftest)
  # The same selftest with $TMPDIR pointed INSIDE a build tree, which is where this repository's
  # own guidance puts it (`prosper/build-linux`). The selftest writes its fixtures into $TMPDIR, and
  # collect_files used to exclude by substring on the ABSOLUTE path, so that TMPDIR discarded every
  # fixture: twelve positive arms printed `got=[]` under the banner "the scanner's own signatures
  # are broken" while the twelve must-not-match arms passed VACUOUSLY. Three lanes hit it inside
  # five minutes and one read it as a real regression (#2658, #2660, #2661).
  #
  # --require-hostile-tmpdir is what keeps this case honest. `tempfile` falls back to /tmp without
  # a word when $TMPDIR names a directory that does not exist, so a case that merely SETS the
  # variable would go green on an environment it never had -- an arm that cannot show its lever
  # moved. The flag asserts the temp root really is a build-tree path before any arm runs.
  set(PROSPER_DIAG_GATE_HOSTILE_TMPDIR "${CMAKE_CURRENT_BINARY_DIR}/build-hostile-tmpdir")
  file(MAKE_DIRECTORY "${PROSPER_DIAG_GATE_HOSTILE_TMPDIR}")
  add_test(NAME diag_gate_selftest_hostile_tmpdir
           COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/env/check_diag_gates.py"
                   --selftest --require-hostile-tmpdir)
  set_tests_properties(diag_gate_selftest_hostile_tmpdir PROPERTIES
                       ENVIRONMENT "TMPDIR=${PROSPER_DIAG_GATE_HOSTILE_TMPDIR}")
  # Most cases here assert on what must NOT appear (negative time, failed dispatches summed in, an
  # image share computed against a filtered denominator, a per-binding mean over aliased folds),
  # because each is a table this tool has already produced once. Note what this gate does NOT do: it
  # feeds the tool SYNTHETIC records, so it pins the tool against its own model and can never notice
  # that model drifting from live_compute.cpp. Only the tool's runtime broken_sum/broken_nest warning
  # detects real drift, and only when someone reads stderr.
  add_test(NAME compute_phase_report_logic
           COMMAND "${Python3_EXECUTABLE}"
                   "${CMAKE_SOURCE_DIR}/tools/perf/test_compute_phase_report.py")
  # The audio-delivery and flip-pacing reports, recovered from #2984 (#3033). Registered rather
  # than left as scripts with a --self-test nobody runs: both shipped with working self-tests in
  # that PR and neither was wired into ctest, which is a test that cannot go red. Same synthetic-
  # record caveat as compute_phase_report_logic above — these pin each tool against its own model
  # of the log format, so they catch a parser regression and cannot notice the EMITTER drifting.
  add_test(NAME audio_delivery_report_logic
           COMMAND "${Python3_EXECUTABLE}"
                   "${CMAKE_SOURCE_DIR}/tools/perf/test_audio_delivery_report.py")
  add_test(NAME flip_pacing_report_logic
           COMMAND "${Python3_EXECUTABLE}"
                   "${CMAKE_SOURCE_DIR}/tools/perf/test_flip_pacing_report.py")
  # compare_runs.py: refuses two screenshot manifests whose run conditions differ. Synthetic
  # manifests; every refusal arm differs from the matching pair in exactly one condition.
  add_test(NAME compare_runs_logic
           COMMAND "${Python3_EXECUTABLE}"
                   "${CMAKE_SOURCE_DIR}/tools/perf/test_compare_runs.py")
  # The two /proc samplers (#3400 / #3401). Both run against a /proc tree the test BUILDS, so they
  # need no target process, no privileges and no sleeping: wait_profile's arms pin per-thread
  # identity (same-name threads, TID reuse, renames, PID reuse), and lock_holder's pin that a run
  # which could read no thread's syscall state is UNAVAILABLE (exit 2) rather than a negative.
  add_test(NAME wait_profile_logic
           COMMAND "${Python3_EXECUTABLE}"
                   "${CMAKE_SOURCE_DIR}/tools/perf/test_wait_profile.py")
  add_test(NAME lock_holder_logic
           COMMAND "${Python3_EXECUTABLE}"
                   "${CMAKE_SOURCE_DIR}/tools/perf/test_lock_holder.py")
  # The queue-depth timeline reader (#3033). Registered for a reason specific to this one: its
  # verdict is a DISCRIMINATOR that a pacer A/B rests on, and its first version could not tell a
  # port that was starved from a port that was merely silent -- both read queued=0. It reported a
  # 600 s Blasphemous 2 run as 53% dry before that gate existed. The arms pin all three conditions
  # of the gate, each of which was found untested by mutating it away.
  add_test(NAME audio_queue_timeline_report_logic
           COMMAND "${Python3_EXECUTABLE}"
                   "${CMAKE_SOURCE_DIR}/tools/perf/test_audio_queue_timeline_report.py")
  # The shader-conformance scanner (#3036), registered because its whole value is a CLEAN result
  # and its original failure mode was manufacturing one: a capture that yielded zero examined
  # shaders exited 0, and findings could be attributed to a capture whose dumps had all failed.
  # It drives the scanner end to end against a stub gpu_replay and a `spirv-dis` shim, so it needs
  # neither a GPU nor the Vulkan toolchain, and it pins the exit contract (0 clean / 1 mismatch /
  # 2 could-not-scan) rather than only the two parsers. Ten mutations of the logic it covers,
  # including `DIM_ARRAYED` dropping the 2D_ARRAY case that is the tool's entire reason to exist,
  # each turn it red.
  #
  # It needs the BUILT `shader_inspect` since #3184: the scanner's MIMG walk is now prosper's own
  # RDNA2 decoder rather than a Python port of its length rules, so the decode arms exercise
  # `rdna2_decode.cpp` itself and a change to its length dispatch reddens this suite. Guarded on
  # `TARGET prosper_core` because that is what governs the executable; the generator expression
  # resolves against the target added later in this file. Without the binary the suite exits 2
  # (could not test) rather than reporting a pass.
  if(TARGET prosper_core)
    add_test(NAME shader_conformance_scan_logic
             COMMAND "${Python3_EXECUTABLE}"
                     "${CMAKE_SOURCE_DIR}/tools/shader_conformance/scan.py" "--self-test"
                     "--mimg-decoder" "$<TARGET_FILE:shader_inspect>")
  endif()
  # stack_profile's frame classifier, against RECORDED gdb output. Deliberately needs no gdb and no
  # ptrace: an end-to-end variant would skip silently wherever ptrace is unavailable, and a silently
  # skipped test is indistinguishable from a passing one. What it pins is the case that already
  # produced a wrong answer once — a versioned glibc symbol (`pthread_cond_wait@@GLIBC_2.3.2`) that
  # defeated the `$`-anchored plumbing patterns, so a libc internal was named as the application's
  # blocking site. Reverting only the version-stripping fix fails four of these cases.
  add_test(NAME stack_profile_logic
           COMMAND "${Python3_EXECUTABLE}"
                   "${CMAKE_SOURCE_DIR}/tools/perf/test_stack_profile.py")
  add_test(NAME perf_caller_report
           COMMAND "${Python3_EXECUTABLE}"
                   "${CMAKE_SOURCE_DIR}/tools/perf/test_perf_caller_report.py")
  if(GIT_EXECUTABLE)
    # The provenance GATE: refuses to quote a measurement from a build that is not the revision the
    # author thinks it is. Pure argument/exit-status contract, no build required — but it resolves
    # real refs and creates a scratch repository, so it needs git exactly as its sibling below does.
    add_test(NAME check_build_revision
             COMMAND "${Python3_EXECUTABLE}"
                     "${CMAKE_SOURCE_DIR}/tools/revision/test_check_build_revision.py")
    add_test(NAME build_revision_refresh
             COMMAND "${Python3_EXECUTABLE}"
               "${CMAKE_SOURCE_DIR}/tools/revision/test_build_revision.py"
               --cmake "${CMAKE_COMMAND}"
               --git "${GIT_EXECUTABLE}"
               --module "${CMAKE_SOURCE_DIR}/cmake/ProsperBuildRevision.cmake"
               --scratch-root "${CMAKE_CURRENT_BINARY_DIR}/build-revision-regression"
               --generator "${CMAKE_GENERATOR}")
    add_test(NAME compiler_source_identity
             COMMAND "${Python3_EXECUTABLE}"
               "${CMAKE_SOURCE_DIR}/tools/revision/test_source_identity.py"
               --cmake "${CMAKE_COMMAND}"
               --module "${CMAKE_SOURCE_DIR}/cmake/ProsperBuildRevision.cmake"
               --generator "${CMAKE_GENERATOR}")
  endif()
endif()
# Path to the (gitignored) local game dump; override with -DGAME_DUMP=...
#
# The relative fallback resolves to `<source>/../PPSA24651-app0`, which inside a git worktree is a
# directory no dump lives in — that is instrument trap 126, where a build silently runs a SMALLER
# suite and still prints "100% tests passed" with exit 0, because the percentage is of what ran.
# Preferring $PROSPER_GAME_ROOT makes the common case resolve on its own without putting anybody's
# local directory layout in this file. `-DGAME_DUMP=` still wins: a CACHE entry that already exists
# is not overwritten by set(... CACHE ...).
if(DEFINED ENV{PROSPER_GAME_ROOT})
  set(GAME_DUMP "$ENV{PROSPER_GAME_ROOT}/PPSA24651-app0" CACHE PATH "Local PS5 game dump root")
else()
  set(GAME_DUMP "${CMAKE_SOURCE_DIR}/../PPSA24651-app0" CACHE PATH "Local PS5 game dump root")
endif()
set(HAVE_GAME_DUMP OFF)
if(EXISTS "${GAME_DUMP}/eboot.bin")
  set(HAVE_GAME_DUMP ON)
else()
  # Loud, and says how to fix it: this message is the ONLY announcement, it happens once at configure
  # time, and every later build inherits the reduced suite in silence.
  message(WARNING "Game dump not found at ${GAME_DUMP} — dump-backed tests will be SKIPPED, so "
                  "ctest will report a smaller count and still exit 0. Set PROSPER_GAME_ROOT or "
                  "pass -DGAME_DUMP=<DUMP_ROOT>/PPSA24651-app0 if you did not intend this.")
endif()

# Which title `GAME_DUMP` actually points at (#1573). A handful of ctest cases assert against ONE
# title's bytes — import counts, an IL2CPP module layout, that title's embedded shader blobs — so
# pointing GAME_DUMP at a different game turns them red for a reason nothing in the ctest output
# names. A bring-up agent's first action on a new title is exactly that configure, so the red run is
# the first thing they see, and every plausible reading of it (master is broken / this dump is
# corrupt / bisect) is wrong.
#
# `sce_sys/param.json`'s `titleId` is authoritative; the `PPSAxxxxx-app0` directory name is a second,
# independent source for a dump laid out without `sce_sys/`. If NEITHER yields an id the title-gated
# cases are registered and RUN as before — a detection failure must stay fail-visible rather than
# converting three real guards into silent skips, which is the defect this whole area is about.
# CONFIDENCE: HIGH (both sources read directly from every dump on the dev box).
set(GAME_DUMP_TITLE_ID "")
if(HAVE_GAME_DUMP)
  if(EXISTS "${GAME_DUMP}/sce_sys/param.json")
    file(READ "${GAME_DUMP}/sce_sys/param.json" prosper_param_json)
    string(JSON prosper_param_title_id ERROR_VARIABLE prosper_param_error
           GET "${prosper_param_json}" "titleId")
    if(NOT prosper_param_error)
      set(GAME_DUMP_TITLE_ID "${prosper_param_title_id}")
    endif()
    unset(prosper_param_json)
  endif()
  if(GAME_DUMP_TITLE_ID STREQUAL "")
    get_filename_component(prosper_dump_dir_name "${GAME_DUMP}" NAME)
    if(prosper_dump_dir_name MATCHES "^([A-Z]+[0-9]+)")
      set(GAME_DUMP_TITLE_ID "${CMAKE_MATCH_1}")
    endif()
  endif()
  if(GAME_DUMP_TITLE_ID STREQUAL "")
    message(STATUS "Game dump title id could not be determined from ${GAME_DUMP} — "
                   "title-specific tests will run against it")
  else()
    message(STATUS "Game dump title id: ${GAME_DUMP_TITLE_ID}")
  endif()
endif()

# Register `name` as a ctest case that is SKIPPED with `reason`, rather than not registered at all.
# `cmake -E echo` prints the reason and exits 0; SKIP_REGULAR_EXPRESSION is what turns that into a
# `(Skipped)` result. The point is that the case still occupies a line in the ctest summary: a test
# that silently disappears from the registered set costs coverage with no line to read and no count
# to compare against (the failure mode `docs/process/VERIFICATION.md` § CI records for the Vulkan flag).
function(prosper_add_skipped_test name reason)
  add_test(NAME ${name} COMMAND ${CMAKE_COMMAND} -E echo "SKIPPED: ${reason}")
  set_tests_properties(${name} PROPERTIES SKIP_REGULAR_EXPRESSION "SKIPPED: ")
endfunction()

# Register a ctest case that asserts against ONE title's content. Runs normally when GAME_DUMP is
# that title (or when no title id could be derived); otherwise registers a visible skip naming both
# the required title and the configured one, so the reason is in the run instead of in a doc.
function(prosper_add_title_gated_test name title)
  if(GAME_DUMP_TITLE_ID STREQUAL "${title}" OR GAME_DUMP_TITLE_ID STREQUAL "")
    add_test(NAME ${name} COMMAND ${ARGN})
  else()
    prosper_add_skipped_test(${name}
      "${name} asserts against ${title}'s own content; GAME_DUMP is configured with \
${GAME_DUMP_TITLE_ID}. Re-run cmake with -DGAME_DUMP=<dump root>/${title}-app0 to exercise it.")
    # Also say it here. ctest prints `(Skipped)` but not the reason unless it is run with `-V`, and
    # configure is the moment the reader chose the dump — so this is where the explanation lands in
    # front of them without their having to know to ask for it.
    message(STATUS
      "ctest case '${name}' will be SKIPPED: it asserts against ${title}'s own content, "
      "GAME_DUMP is ${GAME_DUMP_TITLE_ID}")
  endif()
endfunction()

if(TARGET prosper_core)
  prosper_add_gtest(test_fragment_scalar_reads
    SOURCES tests/gpu/recompiler/test_fragment_scalar_reads.cpp
    INCLUDES tests LIBRARIES prosper_core)
  prosper_add_gtest(test_fragment_entry_prefix
    SOURCES tests/gpu/execute/test_fragment_entry_prefix.cpp
    INCLUDES tests LIBRARIES prosper_core)
  prosper_add_gtest(test_ps_pull_model
    SOURCES tests/gpu/execute/test_ps_pull_model.cpp
    INCLUDES tests LIBRARIES prosper_core)

  # nid_census: every guest import that falls to the dispatcher's return-0 default (#2081).
  # Links prosper_core because both of its inputs must be the real ones — `Module::load` for the
  # imports the loader will actually try to bind, `module_export_nids` for what a sibling module
  # already defines, and the live `Hle` registry for whether a handler exists. A source grep over
  # register_fn() call sites cannot see any of the three.
  add_executable(nid_census tools/nid_census/nid_census.cpp)
  target_link_libraries(nid_census PRIVATE prosper_core)
  if(Python3_Interpreter_FOUND)
    add_test(NAME nid_census_context COMMAND "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/tests/tools/test_nid_census_context.py"
      $<TARGET_FILE:nid_census>)
    set_tests_properties(nid_census_context PROPERTIES TIMEOUT 60)
  endif()

  add_executable(test_module tests/test_module.cpp)
  target_link_libraries(test_module PRIVATE prosper_core)
  if(HAVE_GAME_DUMP)
    # Messenger-specific: the assertions pin PPSA24651's import/segment/relocation counts (#1573).
    prosper_add_title_gated_test(module_loads_eboot PPSA24651
             test_module "${GAME_DUMP}/eboot.bin")
  endif()

  add_executable(test_nid tests/hle/dispatch/test_nid.cpp)
  target_link_libraries(test_nid PRIVATE prosper_core)
  if(HAVE_GAME_DUMP)
    add_test(NAME nid_hash_matches_imports
             COMMAND test_nid "${GAME_DUMP}/eboot.bin")
  endif()

  # test_scratch.h: scratch creation survives a peer removing the shared root (ctest -j race).
  prosper_add_gtest(test_scratch_root_race
    SOURCES tests/misc/test_scratch_root_race.cpp
    INCLUDES tests)

  # Dump-independent golden vectors pin NID hashing in the default unit-test set.
  prosper_add_gtest(test_nid_vectors
    SOURCES tests/hle/dispatch/test_nid_vectors.cpp
    LIBRARIES prosper_core)

  # Split-T# provenance proof (#4197): pure decoder/fold cases, no dump.
  # #4120/#4274: storage-image DST_SEL routing rules (pure).
  prosper_add_gtest(test_storage_dst_sel
    SOURCES tests/gpu/recompiler/test_storage_dst_sel.cpp
    INCLUDES src
    LIBRARIES prosper_core)
  prosper_add_gtest(test_split_t8_dataflow
    SOURCES tests/gpu/recompiler/test_split_t8_dataflow.cpp
    INCLUDES src
    LIBRARIES prosper_core)
  prosper_add_gtest(test_split_t8_loop_defs
    SOURCES tests/gpu/recompiler/test_split_t8_loop_defs.cpp
    INCLUDES src
    LIBRARIES prosper_core)

  prosper_add_gtest(test_split_t8_sop2_width
    SOURCES tests/gpu/recompiler/test_split_t8_sop2_width.cpp
    INCLUDES src
    LIBRARIES prosper_core)
  prosper_add_gtest(test_split_t8_tail_blocks
    SOURCES tests/gpu/recompiler/test_split_t8_tail_blocks.cpp
    INCLUDES src
    LIBRARIES prosper_core)

  # #3529: the initial value of an unresolved DATA import. Pure -- no dump, no mapping.
  prosper_add_gtest(test_import_data_seed
    SOURCES tests/hle/dispatch/test_import_data_seed.cpp
    LIBRARIES prosper_core)

  add_executable(test_hle_registered tests/hle/test_hle_registered.cpp)
  target_link_libraries(test_hle_registered PRIVATE prosper_core)
  add_test(NAME hle_functions_registered COMMAND test_hle_registered)

  # No NID registered twice with a DIFFERENT handler (the #330 shadow class): a later register_*()
  # silently overwriting a real impl with a stub returns garbage to the guest. Pure, no game dump.
  prosper_add_gtest(test_hle_no_shadow
    SOURCES tests/hle/test_hle_no_shadow.cpp
    LIBRARIES prosper_core)

  # Runtime IL2CPP symbolication (#2551): name the C# method containing a live guest PC. The
  # fixture is committed and synthetic — no game dump, no Il2CppDumper — so this never degrades
  # into a silent skip. The binary doubles as the probe the agreement test drives.
  # Tie reporting is a pure CPU resolver contract, independent of boot mappings or graphics.
  prosper_add_gtest(test_il2cpp_ambiguity
    SOURCES tests/host/symbols/test_il2cpp_ambiguity.cpp src/host/symbols/il2cpp_symbols.cpp
    INCLUDES src tests
    LIBRARIES Threads::Threads
    PROPERTIES TIMEOUT 20)

  add_executable(test_il2cpp_symbols tests/host/symbols/test_il2cpp_symbols.cpp)
  target_include_directories(test_il2cpp_symbols PRIVATE tests)
  target_link_libraries(test_il2cpp_symbols PRIVATE prosper_core)
  add_test(NAME il2cpp_symbol_resolution
           COMMAND test_il2cpp_symbols
                   "${CMAKE_CURRENT_SOURCE_DIR}/tests/data/il2cpp_symtab_fixture.symtab")
  # Each synthetic boot owns fixed guest mappings, so run the wiring controls in fresh processes.
  foreach(mode valid oversized absent)
    add_test(NAME il2cpp_symbol_boot_${mode}
             COMMAND test_il2cpp_symbols --boot-bounds ${mode})
    set_tests_properties(il2cpp_symbol_boot_${mode} PROPERTIES TIMEOUT 20)
  endforeach()
  if(Python3_Interpreter_FOUND)
    # ...and the runtime resolver must answer exactly what tools/il2cpp/resolve.py answers. They
    # share the record set by construction (resolve.py emits the table the runtime reads), but the
    # nearest-preceding rule, its window and its tie handling are implemented twice, so this drives
    # both CLIs over one synthesized corpus and requires them to agree address-for-address.
    add_test(NAME il2cpp_symtab_agreement
             COMMAND "${Python3_EXECUTABLE}"
                     "${CMAKE_CURRENT_SOURCE_DIR}/tools/il2cpp/test_symtab_agreement.py"
                     "$<TARGET_FILE:test_il2cpp_symbols>")
  else()
    prosper_add_skipped_test(il2cpp_symtab_agreement
      "il2cpp_symtab_agreement drives tools/il2cpp/resolve.py; no Python 3 interpreter was found.")
  endif()

  # Ground truth for tools/re/hle_handler_map.py: the registry register_builtin_hle() really builds
  # (#2070). Pure setup — no game dump, no GPU — so the static parser can be reconciled against the
  # compiled binary on every run instead of being trusted.
  add_executable(hle_registry_dump tools/re/hle_registry_dump.cpp)
  target_link_libraries(hle_registry_dump PRIVATE prosper_core)

  # Cooperative stop signal (issue #164 P0): request_stop/stop_requested the frontend uses to wind
  # the guest run-loop down on window-close. Pure, no deps.
  prosper_add_gtest(test_lifecycle
    SOURCES tests/host/platform/test_lifecycle.cpp
    LIBRARIES prosper_core)

  # GPU submit gate (#3225): the counted region + bounded drain prosper-app runs before std::_Exit,
  # so a detached guest thread cannot be killed inside an amdgpu submission (which parks the task in
  # __drm_exec_lock_obj and takes the host compositor down with it). Pure — no Vulkan, no dump — so
  # the admission/refusal/drain-timeout contract is asserted on every host, including CI without a
  # GPU. Its own timing assertions are one-sided (a lower bound on a timeout, an upper bound far
  # from the deadline), so it does not go red under host load.
  add_executable(test_gpu_submit_gate tests/host/platform/test_gpu_submit_gate.cpp)
  target_link_libraries(test_gpu_submit_gate PRIVATE prosper_core)
  add_test(NAME gpu_submit_gate COMMAND test_gpu_submit_gate)

  # Screenshot evidence classification/JSON is pure and must remain testable without Vulkan or a dump.
  add_executable(test_capture_manifest tests/test_capture_manifest.cpp
                                       tools/screenshot/capture_manifest.cpp
                                       # The framerate the manifest reports. A pure TU with no
                                       # dependency beyond the standard library, so it links here
                                       # without dragging prosper_core (or Vulkan) into the test.
                                       src/gpu/present/present_frame_rate.cpp)
  # `src` and the revision library are for `build_revision.hpp` only: the run header records the
  # revision the binary was compiled from, so the manifest self-describes. Still no Vulkan, no dump.
  target_include_directories(test_capture_manifest PRIVATE tools/screenshot src)
  target_link_libraries(test_capture_manifest PRIVATE prosper_build_revision)
  add_test(NAME capture_manifest COMMAND test_capture_manifest)

  # The burned-in fps annotation (`screenshot --fps-overlay`). Rasterizing text into an RGBA buffer
  # is pure, and an overlay that silently draws nothing looks exactly like one that was never asked
  # for -- so it is asserted on pixels, here, without a device.
  prosper_add_gtest(test_overlay_text
    SOURCES tests/test_overlay_text.cpp tools/screenshot/overlay_text.cpp
    INCLUDES tools/screenshot)

  # va2foff resolution order (issue #113): DYNAMIC/PROCPARAM/TLS must never shadow
  # the PT_LOAD that truly maps a VA. Pure unit test — no game dump required.
  add_executable(test_va2foff tests/test_va2foff.cpp)
  target_link_libraries(test_va2foff PRIVATE prosper_core)
  add_test(NAME va2foff_load_priority COMMAND test_va2foff)

  # SELF/ELF/loader parsing must handle a malformed/truncated dump without an out-of-bounds read:
  # overflow-safe rd() bounds (self_read_ok), NUL-terminated str_at, and the linker init-array p+8
  # guard. Pure, no game dump.
  add_executable(test_self_bounds tests/test_self_bounds.cpp)
  target_link_libraries(test_self_bounds PRIVATE prosper_core)
  add_test(NAME self_bounds COMMAND test_self_bounds)

  # SELF data-segment key masks flag bits >= 32 (#339): the SCE segment id is bits [31:20]; without
  # the mask a data segment keys the map wrong and every phdr lookup misses. Pure, no game dump.
  add_executable(test_self_segment_key tests/test_self_segment_key.cpp)
  target_link_libraries(test_self_segment_key PRIVATE prosper_core)
  add_test(NAME self_segment_key COMMAND test_self_segment_key)

  # sceKernelNanosleep's GUEST timespec contract (#3013). Registered OUTSIDE the if(UNIX) block
  # below ON PURPOSE: what it pins is a WINDOWS defect -- MinGW-w64 declares a 32-bit `long tv_nsec`
  # while the guest timespec is FreeBSD x86-64's 64-bit field, so writing the remainder through a
  # host struct covered 12 of 16 bytes -- and a Unix-only registration would build it everywhere
  # except the platform it is about. The second property, that a malformed tv_nsec returns promptly
  # rather than sleeping ~2 s or ~292 years, is platform-independent. Asserts promptness, not
  # accuracy: sleep precision belongs to host::sleep_until_steady_ns and a wall-clock assertion on
  # it would flake on a loaded host.
  add_executable(test_kernel_nanosleep tests/hle/kernel/test_kernel_nanosleep.cpp)
  target_link_libraries(test_kernel_nanosleep PRIVATE prosper_core)
  add_test(NAME kernel_nanosleep COMMAND test_kernel_nanosleep)

  # sceKernelRaiseException under a stop-the-world burst (#4324). POSIX only: Windows delivers
  # through its own suspend/redirect path (test_win_exception_delivery). The macOS pending table
  # was 16 slots and Silksong's ~24-thread IL2CPP burst overflowed it; the test blocks its targets'
  # signals so the overflow is forced, not left to scheduling.
  if(NOT WIN32)
    prosper_add_gtest(test_exc_raise_burst
      SOURCES tests/hle/kernel/test_exc_raise_burst.cpp
      INCLUDES src
      LIBRARIES prosper_core)
  endif()

  # The guest semaphore timed wait (#3013). Platform-neutral like the nanosleep case above and for
  # the same reason: the fix is Windows-only, so gating on UNIX would build the test everywhere
  # except where the defect is.
  #
  # It asserts the MECHANISM -- which primitive served the wait, via sleep_backend_name() -- rather
  # than a wall-clock duration, following test_videoout.cpp and what test_kernel_nanosleep.cpp's
  # header asks for. The first version asserted durations only and review showed NONE of its arms
  # could redden on the pre-fix code: its 40 ms ceiling sat 2.6x above the 15.57 ms defect, and its
  # other two arms measured a POSTED semaphore, which unfixed winpthreads serves in 0.01 ms -- better
  # than the poll loop, so they preferred the defect. A mechanism assertion cannot be satisfied by a
  # lucky duration and does not flake under load. Mutation-checked: reverting to the winpthreads
  # delegation reddens the MECHANISM arm and prints "via none".
  #
  # It used to say "reddens two arms", the second being a wall-clock ceiling, and that ceiling is
  # gone: it would have PASSED the defect, and it also flaked on macOS/Rosetta. The mechanism
  # arm is the discriminator; the bounds that remain are a stub/unit guard.
  #
  # The measurements and the argument live in ONE place, the test file's own header. They were
  # briefly in both, with two different lists -- and a third Rosetta figure existed in neither
  # (23.22 ms, found in a CI log during review). Two copies of a figure list is one copy too
  # many, and this is the copy that gets read second.
  add_executable(test_kernel_sem_timedwait tests/hle/kernel/test_kernel_sem_timedwait.cpp)
  target_link_libraries(test_kernel_sem_timedwait PRIVATE prosper_core)
  add_test(NAME kernel_sem_timedwait COMMAND test_kernel_sem_timedwait)

  # scePthreadCondTimedwait resolves the CONDVAR'S clock for its relative microsecond timeout
  # (#3056), which its POSIX sibling already did and this spelling did not. CROSS-PLATFORM, and
  # deliberately so: the defect is in the shared HLE body, not in either platform's wait primitive,
  # and the discriminator (SCE_KERNEL_CLOCK_VIRTUAL is process CPU time, so an idle wait cannot
  # reach its deadline) reads a RETURN VALUE rather than a duration. See the test's own header for
  # why realtime and monotonic clocks cannot discriminate anything here.
  add_executable(test_kernel_cond_timedwait_clock tests/hle/kernel/test_kernel_cond_timedwait_clock.cpp)
  target_link_libraries(test_kernel_cond_timedwait_clock PRIVATE prosper_core)
  add_test(NAME kernel_cond_timedwait_clock COMMAND test_kernel_cond_timedwait_clock)
  # Bounded: the arm parks a worker on a condvar whose deadline an idle process cannot reach, and
  # relies on the main thread's signal to release it. A regression that loses the signal would
  # otherwise hang the suite rather than fail it.
  set_tests_properties(kernel_cond_timedwait_clock PROPERTIES TIMEOUT 60)
  # Bounded for the same reason as the nanosleep case: the fixed path returns in ~5 ms, so a
  # regression that reverts to a blocking wait on a semaphore nothing will ever post would hang
  # rather than fail, and CTest's default 1500 s hang reads as neither pass nor fail.
  set_tests_properties(kernel_sem_timedwait PROPERTIES TIMEOUT 60)

  # scePthreadSemInit must forward a real sem_init() failure instead of discarding it and
  # manufacturing success (#3068). Platform-neutral: it probes the host sem_init() with an
  # out-of-range value first and only asserts the guest-visible contract when the host call
  # actually failed, so it is meaningful on every CI platform (EINVAL on Linux, ENOSYS on Darwin)
  # without needing per-platform code, and skips itself explicitly rather than passing vacuously
  # if some future libc happens to accept the probe value.
  prosper_add_gtest(test_kernel_sem_init_error
    SOURCES tests/hle/kernel/test_kernel_sem_init_error.cpp
    LIBRARIES prosper_core)
  # Bounded because arm 3 guards a ~292-YEAR sleep: without a timeout a regression there presents as
  # CTest hanging for its default 1500 s rather than as a failure, and a hang is the one outcome nobody
  # reads as a red test. Generous against a loaded host -- what it catches is seconds-to-forever.
  set_tests_properties(kernel_nanosleep PROPERTIES TIMEOUT 60)

  # LABEL: `timing-margin` -- the two tests above whose green verdict rests on a MEASURED wall-clock
  # margin, and which print that margin on stdout so it can be quoted.
  #
  # #3067 is why this label exists. A red `macOS (x86_64 / Rosetta 2)` job was recorded as
  # "unresolved, two candidates" while the failing job's own log already named the arm and the
  # number -- `[FAIL] and returned inside one winpthreads tick of the request, not after it` /
  # `(timeout took 13.58 ms via none)` against a 12 ms ceiling. The number was there because the
  # test prints it; it was only in the log because that run was RED. `--output-on-failure` discards
  # it from every green run, so the runs on either side of a near-miss carry no evidence at all and
  # the headroom cannot be checked without reddening the job first -- the #3208 defect (a threshold
  # whose margin nobody has measured recently) with the measurement one flag away.
  #
  # The CI jobs exclude this label from the parallel suite (`-LE timing-margin -j4`, so a margin is
  # not measured beside other load) and run it serially with `-V` instead, so every log carries the
  # figures.
  # `kernel_nanosleep` is deliberately NOT labelled: its wall-clock bounds are 500 ms order-of-
  # magnitude guards against a ~292-year sleep and it prints no measured value, so a verbose re-run
  # would report nothing. Add the label when a test both DEPENDS ON and PRINTS a margin.
  set_tests_properties(kernel_sem_timedwait kernel_cond_timedwait_clock
                       PROPERTIES LABELS "timing-margin")

  # Guest static-TLS Variant II layout uses each module's real PT_TLS p_align (#143). The layout
  # code is Linux-only (guest %fs), so gate the test on it too.
  if(UNIX)
    prosper_add_gtest(test_guest_tls_align
      SOURCES tests/loader/test_guest_tls_align.cpp
      LIBRARIES prosper_core)
    # #3635: a second guest_tls_activate_thread() on one thread keeps its guest TCB instead of
    # installing a fresh zeroed block. The wrfsbase path is x86-64 Linux only; the test skips elsewhere.
    add_executable(test_guest_tls_reactivate tests/host/tls/test_guest_tls_reactivate.cpp)
    target_link_libraries(test_guest_tls_reactivate PRIVATE prosper_core)
    add_test(NAME guest_tls_reactivate COMMAND test_guest_tls_reactivate)
  endif()

  # Cross-module TLS resolution (#136/#338): dynamic and initial-exec relocations resolve through
  # the DEFINING module's id, symbol offset, and shared static layout. Pure synthetic unit test.
  prosper_add_gtest(test_dtpmod_tls
    SOURCES tests/test_dtpmod_tls.cpp
    LIBRARIES prosper_core)

  # AMD SSE4a INSERTQ/EXTRQ emulation (the SIGILL handler emulates these Zen2-only ops on Intel
  # hosts). Golden bit-math vectors — header-only, no game dump required.
  # Does not link prosper_core, so it does not inherit its PUBLIC `src`.
  prosper_add_gtest(test_sse4a SOURCES tests/host/x86/test_sse4a.cpp INCLUDES src)

  # macOS/Rosetta %fs trap-emulation decode core (guest-TLS access redirection). Golden
  # instruction vectors incl. the #727 no-REX AH/CH/DH/BH byte-register rules — header-only,
  # runs on Linux where the __APPLE__ handler itself never compiles.
  # Does not link prosper_core, so it does not inherit its PUBLIC `src`.
  prosper_add_gtest(test_fs_emu SOURCES tests/host/tls/test_fs_emu.cpp INCLUDES src)

  # Signal-handler-safe raw syscalls (src/host/platform/raw_syscall.hpp): the fault_handler's
  # PROSPER_NULL_PAGE backing must not touch errno/TLS on guest-%fs threads (issue #1071).
  if(UNIX AND NOT APPLE)
    # Does not link prosper_core, so it does not inherit its PUBLIC `src`.
    add_executable(test_raw_syscall tests/host/platform/test_raw_syscall.cpp)
    target_include_directories(test_raw_syscall PRIVATE src)
    target_include_directories(test_raw_syscall PRIVATE tests)
    add_test(NAME raw_syscall COMMAND test_raw_syscall)
  endif()

  # sceAudioOut HLE: format decode, port lifecycle, PCM forwarding, volume, batch, error paths
  # (via a capturing AudioSink — headless, no device).
  add_executable(test_audio tests/hle/audio/test_audio.cpp)
  target_link_libraries(test_audio PRIVATE prosper_core)
  add_test(NAME audio_hle COMMAND test_audio)
  if(Python3_Interpreter_FOUND)
    add_test(NAME audio_flow_publication
             COMMAND "${Python3_EXECUTABLE}"
                     "${CMAKE_SOURCE_DIR}/tests/hle/audio/test_audio_flow_publication.py"
                     $<TARGET_FILE:test_audio>)
    add_test(NAME audio_lifecycle_publication
             COMMAND "${Python3_EXECUTABLE}"
                     "${CMAKE_SOURCE_DIR}/tests/hle/audio/test_audio_lifecycle.py"
                     $<TARGET_FILE:test_audio>)
  endif()

  # NGS2 waveform-block streaming (#1059/#1060 part 3): a fed block makes the voice Playing and advances
  # num_decoded_samples across renders (the Dead Cells music stream clock). POSIX-gated (needs a fixed
  # guest-address PCM mapping); the lifecycle assertions run everywhere.
  prosper_add_gtest(test_ngs2_streaming
    SOURCES tests/hle/test_ngs2_streaming.cpp
    LIBRARIES prosper_core)

  # libSceAjm (#187): headless audio-decode lifecycle — handle out-params filled with valid values
  # (context/instance/batch), inputs validated, batch reports complete. Pure, no game dump.
  prosper_add_gtest(test_ajm
    SOURCES tests/hle/test_ajm.cpp
    LIBRARIES prosper_core)

  prosper_add_gtest(test_avplayer
    SOURCES tests/gpu/texture/test_avplayer.cpp
    LIBRARIES prosper_core)

  # #2270. Three ctest entries over ONE binary, because each arm's environment variable is read once
  # into a function-local static (deliberately — this is a per-frame path), so the arms cannot share
  # a process.
  add_executable(test_videodec2_decode tests/hle/video/test_videodec2_decode.cpp)
  # Includes a shared fixture, which resolves from the `tests` root.
  target_include_directories(test_videodec2_decode PRIVATE tests)
  target_link_libraries(test_videodec2_decode PRIVATE prosper_core)
  add_test(NAME videodec2_decode COMMAND test_videodec2_decode)
  add_test(NAME videodec2_decode_optout COMMAND test_videodec2_decode --no-decode)
  add_test(NAME videodec2_decode_format COMMAND test_videodec2_decode --format-probe)

  # #3669. libSceVdecsw software video decoder test. Verifies 0x50 config adaptation,
  # asynchronous FIFO AU staging, staging-time memory copying, pictureCount offset (+0x0B),
  # TrySyncDecodeInput zeroing, and lifecycle management (Reset / DeleteDecoder).
  prosper_add_gtest(test_vdecsw_decode
    SOURCES tests/hle/video/test_vdecsw_decode.cpp
    INCLUDES tests
    LIBRARIES prosper_core)

  # AGC's colour-block utility pixel programs, recognised so a DCC_DECOMPRESS helper never runs as
  # colour (Silksong's variant exported 1.0 and painted its boot screens white). Pure decoding.
  prosper_add_gtest(test_dcc_helper_program
    SOURCES tests/gpu/execute/test_dcc_helper_program.cpp
    INCLUDES src)

  # #2898. The SPS/VUI parser and the picture-info fill are pure host-side code exercised
  # against hand-built SPS bitstreams written by an independent bit-writer in the test.
  # The fill layout asserts every offset the guest has been observed reading, and carries
  # a mutation arm (absent-meta fill over a poisoned buffer) so removing the fill cannot
  # pass vacuously.
  prosper_add_gtest(test_h264_sps
    SOURCES tests/hle/video/test_h264_sps.cpp
    LIBRARIES prosper_core)

  add_executable(test_json2 tests/hle/test_json2.cpp)
  target_link_libraries(test_json2 PRIVATE prosper_core)
  add_test(NAME json2_hle COMMAND test_json2)

  prosper_add_gtest(test_equeue_events
    SOURCES tests/hle/test_equeue_events.cpp
    LIBRARIES prosper_core)

  # APR completion delivery on a shared equeue (#1673): the constant-tag (CRI) dialect must not be
  # coalesced away, while the counter dialect (#208) and genuine level sources must still coalesce.
  prosper_add_gtest(test_apr_equeue_completion
    SOURCES tests/hle/test_apr_equeue_completion.cpp
    LIBRARIES prosper_core)

  # #3554: all three AMPR userdata registration routes and actual completion/accessor delivery.
  # The legacy sizing route registers only on POSIX; Windows' size query is intentionally pure.
  if(NOT WIN32)
    add_executable(test_ampr_udata tests/hle/kernel/test_ampr_udata.cpp)
    target_link_libraries(test_ampr_udata PRIVATE prosper_core)
    add_test(NAME ampr_udata COMMAND test_ampr_udata)
    set_tests_properties(ampr_udata PROPERTIES TIMEOUT 20)
  endif()

  # #4103: force a real equeue allocation to reuse its handle and verify APR lifetime isolation.
  prosper_add_gtest(test_apr_equeue_lifetime
    SOURCES tests/hle/kernel/test_apr_equeue_lifetime.cpp
    LIBRARIES prosper_core
    PROPERTIES TIMEOUT 20)

  # Variadic libc formatters (issue #122): real va_list capture so %f/XMM + stack + %s-past-4th
  # args format correctly (was a 3-integer-register forward). Cross-platform (host libc printf).
  prosper_add_gtest(test_printf
    SOURCES tests/hle/test_printf.cpp
    LIBRARIES prosper_core)

  # The guest's printf/scanf/strtod family through the REGISTERED handlers (guest-ABI variadic
  # capture, guest va_list hand-off, declared xmm0 returns), against N1570 7.21.6 / 7.22.1.3.
  # Pure, no game dump.
  prosper_add_gtest(test_hle_printf_family
    SOURCES tests/hle/test_hle_printf_family.cpp
    LIBRARIES prosper_core)
  prosper_add_gtest(test_hle_scanf_family
    SOURCES tests/hle/test_hle_scanf_family.cpp
    LIBRARIES prosper_core)

  # Windows' aligned allocator family must remain paired end-to-end. Mixing _aligned_malloc with
  # plain free corrupts the heap under aligned game-buffer churn; ordinary and aligned guest
  # allocations intentionally share one compatible family on that host.
  add_executable(test_libc_alloc tests/hle/test_libc_alloc.cpp)
  target_link_libraries(test_libc_alloc PRIVATE prosper_core)
  target_include_directories(test_libc_alloc PRIVATE tests)
  add_test(NAME libc_alloc COMMAND test_libc_alloc)

  # EventFlag/Sema delete-under-waiter safety (issue #104): delete wakes parked waiters with EACCES
  # and defers the free to the last one out, instead of free()ing the object under a live wait (UAF).
  prosper_add_gtest(test_sync_delete
    SOURCES tests/hle/kernel/test_sync_delete.cpp
    LIBRARIES prosper_core)

  # sceKernelGetOperationMode, the coredump-handler pair and the PRT aperture pair are registered and
  # honour their contracts (out-parameters written, slot range checked) rather than being logged-0
  # stubs.
  prosper_add_gtest(test_kernel_system_stubs
    SOURCES tests/hle/kernel/test_kernel_system_stubs.cpp
    INCLUDES src
    LIBRARIES prosper_core)

  # scePthreadAttrGetstack was unregistered: both outputs were left untouched.
  prosper_add_gtest(test_attr_getstack
    SOURCES tests/hle/kernel/test_attr_getstack.cpp
    INCLUDES src
    LIBRARIES prosper_core)

  # APR status/offset uses 8 bytes and the separate submit ID uses 4; widening the ID
  # overwrites the adjacent guest flag/canary (#4138, #4042).
  prosper_add_gtest(test_apr_result_slot_width
    SOURCES tests/hle/kernel/test_apr_result_slot_width.cpp
    INCLUDES src
    LIBRARIES prosper_core)

  prosper_add_gtest(test_apr_submission
    SOURCES tests/hle/kernel/test_apr_submission.cpp
    INCLUDES src tests
    LIBRARIES prosper_core)

  # lseek/fseek/ftell must address offsets >= 2 GiB. MinGW's long/off_t are 32 bits, so Black Flag's
  # 30 GiB DataPS5_boot.forge allocation-table seek failed and the title aborted itself.
  # An unsupported sparse/sizing filesystem is reported with GTEST_SKIP (ctest "Skipped"), not success.
  prosper_add_gtest(test_large_offset_seek
    SOURCES tests/hle/kernel/test_large_offset_seek.cpp
    INCLUDES src
    LIBRARIES prosper_core
    PROPERTIES TIMEOUT 30)

  # Empty PollSema returns the encoded EBUSY value that guest drain loops require (#1640).
  prosper_add_gtest(test_kernel_sema_poll
    SOURCES tests/hle/kernel/test_kernel_sema_poll.cpp
    LIBRARIES prosper_core)

  # Anchored wall-clock sources (#92): CLOCK_REALTIME ≈ host now, one shared epoch across
  # time()/gettimeofday/sceRtc*, FreeBSD clockid mapping, valid LocalTime calendar dates.
  prosper_add_gtest(test_wall_clock
    SOURCES tests/hle/test_wall_clock.cpp
    LIBRARIES prosper_core)

  # select/pselect pure-sleep shape honours its timeout; a descriptor-set query stays fail-visible
  # (#1660). Asserts elapsed time, not the return value -- the stub this replaces also returned 0.
  add_executable(test_select_sleep tests/hle/kernel/test_select_sleep.cpp)
  target_link_libraries(test_select_sleep PRIVATE prosper_core)
  add_test(NAME select_sleep COMMAND test_select_sleep)
  # This test drives real sleeps through the HLE, and its #3038 arms feed it timeouts a regression
  # would take LITERALLY: carrying an out-of-range tv_nsec into the total makes 0x7fff... a
  # 292-year wait, so the failure mode is a wedge rather than a red. Measured while mutation-testing
  # this PR -- with the range rule removed, test_select_sleep does not return. A bound turns that
  # into a legible timeout instead of ctest's 25-minute default. The whole run is ~0.5 s, so 60 s is
  # a hundredfold headroom and cannot flake under load.
  set_tests_properties(select_sleep PROPERTIES TIMEOUT 60)

  # Opt-in flip-paced monotonic clock (#240): exact frame steps, thread-safe first-flip anchor,
  # and no coupling into CLOCK_REALTIME while virtual time is paused between flips.
  prosper_add_gtest(test_det_clock
    SOURCES tests/hle/test_det_clock.cpp
    LIBRARIES prosper_core)

  # Guest clocks continue through host GPU stalls; only the internal dependency-watchdog clock
  # discounts excess backend time, retaining its hold/resume and concurrent-reader guarantees.
  prosper_add_gtest(test_gpu_time_compensation
    SOURCES tests/hle/kernel/test_gpu_time_compensation.cpp
    LIBRARIES prosper_core)

  # UserService/libkernel startup stubs (HLE audit): getters write deterministic defaults; the
  # libkernel registration/hook calls resolve to benign OK no-ops.
  # Includes a shared fixture, which resolves from the `tests` root.
  prosper_add_gtest(test_service_getters
    SOURCES tests/hle/dispatch/test_service_getters.cpp
    INCLUDES tests
    LIBRARIES prosper_core)

  # #2185: reallocalign was unregistered, so the dispatcher's 0 reached the guest as a NULL
  # allocation — a false FAILURE (reads as OOM), the mirror of the #2081 family below. Asserts the
  # extended alignment survives a resize, which plain realloc never promised. Pure, no game dump.
  prosper_add_gtest(test_reallocalign
    SOURCES tests/hle/test_reallocalign.cpp
    LIBRARIES prosper_core)

  # #2168: a condvar destroy must refuse while a thread is parked inside it.
  prosper_add_gtest(test_cond_destroy_busy
    SOURCES tests/hle/test_cond_destroy_busy.cpp
    LIBRARIES prosper_core)

  # #2178: no Sony pthread spelling may return a bare FreeBSD errno.
  prosper_add_gtest(test_sce_pthread_encoding
    SOURCES tests/hle/test_sce_pthread_encoding.cpp
    LIBRARIES prosper_core)

  # #2168 (barrier half): a barrier destroy must refuse while threads are parked.
  prosper_add_gtest(test_barrier_destroy_busy
    SOURCES tests/hle/test_barrier_destroy_busy.cpp
    LIBRARIES prosper_core)

  prosper_add_gtest(test_reallocf
    SOURCES tests/hle/test_reallocf.cpp
    LIBRARIES prosper_core)

  # Guest libc contracts prosper implements itself (not host forwarders), called through the
  # registered handlers. strlcpy/strlcat are prosper's own code; qsort/bsearch own the guest
  # comparator's calling convention, which differs from the host's on Windows. Pure, no game dump.
  prosper_add_gtest(test_libc_strlcpy
    SOURCES tests/hle/test_libc_strlcpy.cpp
    LIBRARIES prosper_core)
  prosper_add_gtest(test_libc_qsort_bsearch
    SOURCES tests/hle/test_libc_qsort_bsearch.cpp
    LIBRARIES prosper_core)
  # sscanf is a host-convention variadic (not PROSPER_GUEST_ABI); the suite calls it as one.
  prosper_add_gtest(test_libc_sscanf
    SOURCES tests/hle/test_libc_sscanf.cpp
    LIBRARIES prosper_core)

  # #2081 FALSE SUCCESS: an unregistered NID answers the dispatcher's 0, which for an SCE_OK
  # contract is a success report over an out-parameter nothing wrote. Pure, no game dump.
  prosper_add_gtest(test_false_success_nids
    SOURCES tests/hle/test_false_success_nids.cpp
    LIBRARIES prosper_core)

  # #2004: sceAppContentAddcontUnmount releases the claim the mount takes, so a mount/unmount/
  # re-mount cycle works instead of being permanently BUSY. Pure, no game dump.
  # Includes a shared fixture, which resolves from the `tests` root.
  prosper_add_gtest(test_addcontent_unmount
    SOURCES tests/hle/service/test_addcontent_unmount.cpp
    INCLUDES tests
    LIBRARIES prosper_core)

  prosper_add_gtest(test_abnormal_termination_stops
    SOURCES tests/hle/service/test_abnormal_termination_stops.cpp
    INCLUDES tests
    LIBRARIES prosper_core)

  # Includes a shared fixture, which resolves from the `tests` root.
  prosper_add_gtest(test_addcontent_mount_report
    SOURCES tests/hle/service/test_addcontent_mount_report.cpp
    INCLUDES tests
    LIBRARIES prosper_core)

  # libSceSysmodule state query (#2002): IsLoaded answers from prosper's own load history, so a
  # title that gates its initialization on UNLOADED stops skipping that initialization.
  prosper_add_gtest(test_sysmodule_state
    SOURCES tests/hle/test_sysmodule_state.cpp
    LIBRARIES prosper_core)

  # libSceHttp local URI parser (#642): two-pass caller pool sizing, public URI layout, and
  # deterministic error paths. Pure and offline; it never contacts a network.
  prosper_add_gtest(test_http
    SOURCES tests/hle/net/test_http.cpp
    LIBRARIES prosper_core)

  # #2894: libSceHttp2 was unregistered, so a request nobody sent reported SCE_OK and its response
  # getters handed the guest untouched out-parameters.
  prosper_add_gtest(test_http2
    SOURCES tests/hle/net/test_http2.cpp
    LIBRARIES prosper_core)

  add_executable(test_guard_recursion tests/hle/test_guard_recursion.cpp)
  target_link_libraries(test_guard_recursion PRIVATE prosper_core Threads::Threads)
  add_test(NAME guard_recursion COMMAND test_guard_recursion)
  set_tests_properties(guard_recursion PROPERTIES TIMEOUT 30)

  add_executable(test_font tests/hle/test_font.cpp)
  target_link_libraries(test_font PRIVATE prosper_core)
  add_test(NAME font_hle COMMAND test_font)

  add_executable(test_font_render tests/hle/test_font_render.cpp)
  target_link_libraries(test_font_render PRIVATE prosper_core)
  add_test(NAME font_render_hle COMMAND test_font_render)

  add_executable(test_fiber tests/hle/test_fiber.cpp)
  target_link_libraries(test_fiber PRIVATE prosper_core)
  add_test(NAME fiber_hle COMMAND test_fiber)

  # #1603: libSceUlt has no implementation, so it must fail VISIBLY rather than return 0 = SCE_OK.
  # Pure, no game dump.
  prosper_add_gtest(test_ult_fail_visible
    SOURCES tests/hle/test_ult_fail_visible.cpp
    LIBRARIES prosper_core)

  # #1603 Phase 2: the implemented libSceUlt semantics. Mutual exclusion is DEMONSTRATED with real
  # concurrent threads over a non-atomic counter and an exact expected total — a no-op mutex produces
  # a wrong-but-plausible number rather than a crash, so only the exact total is a real check.
  # Pure, no game dump.
  prosper_add_gtest(test_ult_semantics
    SOURCES tests/hle/test_ult_semantics.cpp
    LIBRARIES prosper_core Threads::Threads)

  add_executable(test_posix_sem tests/hle/test_posix_sem.cpp)
  target_link_libraries(test_posix_sem PRIVATE prosper_core)
  add_test(NAME posix_sem_hle COMMAND test_posix_sem)

  # FreeBSD errno numbering across the libkernel 0x80020000|errno family (#1612). Pins the encoding
  # vocabulary, the host->FreeBSD table, and the guest-observable values on the paths a Linux/macOS
  # host can actually reach (sem trywait, timed mutex, wait-on-address, POSIX setcancelstate).
  add_executable(test_sce_errno tests/hle/kernel/test_sce_errno.cpp)
  target_link_libraries(test_sce_errno PRIVATE prosper_core)
  add_test(NAME sce_errno_freebsd COMMAND test_sce_errno)

  # #1659: every diagnostic that prints "<module>+0x<rva>" agrees with the loader's mapped base, and a
  # printed offset round-trips back through PROSPER_BP. Header-only; no dump, no device.
  prosper_add_gtest(test_guest_module_label
    SOURCES tests/host/image/test_guest_module_label.cpp
    LIBRARIES prosper_core)

  # The guest-FS swap stub's saved-base slot is consumed by callback_fs.hpp. Inspect emitted bytes
  # without executing FSGSBASE instructions, and pin both the handler offset and call alignment.
  prosper_add_gtest(test_guest_fs_swap_stub
    SOURCES tests/host/image/test_guest_fs_swap_stub.cpp
    LIBRARIES prosper_core)

  # The frame-pointer walk shared by the fault reporter and PROSPER_HWBP_STACK. It dereferences
  # guest-controlled pointers inside a signal handler, so every stop condition (cycle, unreadable
  # frame, depth cap) is a safety property. Header-only; no dump, no device.
  prosper_add_gtest(test_rbp_chain
    SOURCES tests/host/fault/test_rbp_chain.cpp
    LIBRARIES prosper_core)

  # PlatformUi hook (#347): the app frontend can service the guest's ImeDialog with real UI; with no
  # backend registered the headless auto-complete default is unchanged. Pure, no game dump.
  prosper_add_gtest(test_platform_ui
    SOURCES tests/hle/service/test_platform_ui.cpp
    LIBRARIES prosper_core)

  # SDL3 dialog frontend's guest-struct parsing (#347): pure, SDL-free, so it runs in normal CI even
  # though the SDL_ShowMessageBox glue is display-only. Verifies the MsgDialog/ErrorDialog layouts.
  add_executable(test_dialog_helpers frontends/dialog_sdl3/test_dialog_helpers.cpp
                                     frontends/dialog_sdl3/dialog_helpers.cpp)
                                     # Does not link prosper_core, so it does not inherit its PUBLIC `src`.
                                     target_include_directories(test_dialog_helpers PRIVATE src)
  add_test(NAME dialog_helpers COMMAND test_dialog_helpers)

  # libSceImeDialog lifecycle + libSceSaveData memory API round-trip (#191): auto-completing dialog
  # (no hang) + a real per-(user,slot) Set->Get memory block with verified shadPS4 struct layouts.
  # Includes a shared fixture, which resolves from the `tests` root.
  prosper_add_gtest(test_savedata_ime
    SOURCES tests/hle/test_savedata_ime.cpp
    INCLUDES tests
    LIBRARIES prosper_core)

  # #4275: NP Universal Data System property objects/arrays and SessionSignaling init.
  prosper_add_gtest(test_np_uds_properties
    SOURCES tests/hle/np/test_np_uds_properties.cpp
    LIBRARIES prosper_core)

  # The save's parameter block (#2786): sceSaveDataSetParam/GetParam were unregistered, so both
  # answered SCE_OK while storing and reading nothing. Drives the real NIDs, including a mutation arm
  # (a different block must read back differently) and a persistence arm (it must come off disk).
  prosper_add_gtest(test_savedata_param
    SOURCES tests/hle/test_savedata_param.cpp
    INCLUDES tests
    LIBRARIES prosper_core)

  # sceSaveDataGetMountInfo ignored its mount-point argument (#3653), so a null mount point,
  # "/savedata1", and "/savedata0" after an unmount all got a successful capacity report. Drives the
  # real NIDs -- GetMountInfo, Mount3, Umount -- over a disposable save root, with canaries around
  # the 48-byte output so a write past the ABI reddens.
  prosper_add_gtest(test_savedata_mountinfo
    SOURCES tests/hle/test_savedata_mountinfo.cpp
    INCLUDES tests
    LIBRARIES prosper_core)

  # #3654: the mount wrappers carry the requested blocks into a per-save allocation record, and
  # GetMountInfo reports blocks / freeBlocks from that record and the files in the save. Includes a
  # re-exec of itself to prove the allocation survives a new process.
  add_executable(test_savedata_capacity tests/hle/test_savedata_capacity.cpp)
  target_include_directories(test_savedata_capacity PRIVATE tests)
  target_link_libraries(test_savedata_capacity PRIVATE prosper_core)
  add_test(NAME savedata_capacity COMMAND test_savedata_capacity)

  # The two umount entry points (#3666). sceSaveDataUmount2 read NONE of its arguments -- it
  # unmounted /savedata0 and reported success whatever it was handed, so a title naming another
  # mount point had a live mount discarded on its behalf -- and sceSaveDataUmount answered NOT_FOUND
  # both for a bad argument and for "nothing is mounted", where its siblings answer PARAMETER and
  # NOT_MOUNTED. Drives the real NIDs, including GetEventResult, so a refused umount is shown to
  # queue no completion event. Pins Umount2's argument ORDER (flags first, mount point second),
  # which live capture and four titles' disassembly establish against the obvious reading.
  prosper_add_gtest(test_savedata_umount
    SOURCES tests/hle/test_savedata_umount.cpp
    INCLUDES tests
    LIBRARIES prosper_core)

  # Save data is namespaced by title id (#2734): two titles that pick the same guest save name --
  # `OptionSettings`, or SaveDataMemory's (user 1, slot 0) -- must not share one host save. Runs two
  # application roots over one guest name, because a single-title round trip passes on the broken code.
  prosper_add_gtest(test_savedata_title_namespace
    SOURCES tests/hle/test_savedata_title_namespace.cpp
    INCLUDES tests
    LIBRARIES prosper_core)

  # sceSaveDataDirNameSearch reported NO saves on Windows (#2760): savedata0_list_dirs() was
  # `#ifndef _WIN32`-ed out, so every title's load/continue list was empty there. Drives the real NID
  # over a disposable save root with saves made both through the mount path and by hand, plus a file
  # and a dot-directory that must not be reported. Registered unconditionally so the MinGW job runs it.
  prosper_add_gtest(test_savedata_dirsearch
    SOURCES tests/hle/test_savedata_dirsearch.cpp
    INCLUDES tests
    LIBRARIES prosper_core)

  # libSceIme keyboard API (#186): reports a consistent "no keyboard connected" state (empty resource
  # array + disconnected info, bounds-checked) so the per-frame input poll never reads garbage.
  prosper_add_gtest(test_ime_keyboard
    SOURCES tests/hle/service/test_ime_keyboard.cpp
    LIBRARIES prosper_core)

  # pthread scheduling getters fill their out-param (not a shadowing no-op): scePthreadGetprio/
  # Getaffinity must WRITE priority/mask, else the caller reads uninitialized stack (harmful-stub class).
  add_executable(test_pthread_sched_getters tests/hle/test_pthread_sched_getters.cpp)
  target_link_libraries(test_pthread_sched_getters PRIVATE prosper_core)
  add_test(NAME pthread_sched_getters COMMAND test_pthread_sched_getters)

  prosper_add_gtest(test_pthread_names
    SOURCES tests/hle/test_pthread_names.cpp
    LIBRARIES prosper_core)

  # The ScePthread handle a guest receives must be a readable object, identical across Create/Self,
  # and validated on the way back in. A raw winpthreads index sent AC Black Flag Resynced's main
  # thread into address 3 at eboot+0x161ed91.
  prosper_add_gtest(test_guest_thread_handle
    SOURCES tests/hle/kernel/test_guest_thread_handle.cpp
    INCLUDES src
    LIBRARIES prosper_core
    PROPERTIES TIMEOUT 60)

  add_executable(test_cond_clock tests/hle/test_cond_clock.cpp)
  target_link_libraries(test_cond_clock PRIVATE prosper_core)
  add_test(NAME pthread_cond_clock COMMAND test_cond_clock)

  # Registered on every host, not only under if(UNIX): its _WIN32 arms exercise the Windows
  # ownership registry, which was otherwise covered by no test (noted on #4150).
  # #2619 / #2623: sharing the slot RESOLUTION was not the same as sharing the per-object
  # BOOKKEEPING. Guards the claim-and-poison of a destroyed slot, the missed-wakeup generation,
  # and the cond-destroy waiter count -- through BOTH spellings, since a divergence between them
  # is the defect. Deliberately separate from c11_thread_slots: every arm there passes with the
  # bookkeeping absent. Same generous TIMEOUT, same reason -- one arm parks a real thread.
  add_executable(test_c11_sync_bookkeeping tests/hle/sync/test_c11_sync_bookkeeping.cpp)
  target_link_libraries(test_c11_sync_bookkeeping PRIVATE prosper_core)
  add_test(NAME c11_sync_bookkeeping COMMAND test_c11_sync_bookkeeping)
  set_tests_properties(c11_sync_bookkeeping PROPERTIES TIMEOUT 120)

  # libScePad controller input: ScePadData layout + HostPadState mapping + HLE fill contract (pure).
  # Includes a shared fixture, which resolves from the `tests` root.
  prosper_add_gtest(test_pad
    SOURCES tests/input/test_pad.cpp
    INCLUDES tests
    LIBRARIES prosper_core)

  # Render-cadence inertness: when a retained DMA copy overrides EVERY skip PROSPER_RENDER_EVERY
  # asked for, the run is a cadence-1 run wearing an `every=N` label and must say so (#2837).
  prosper_add_gtest(test_render_cadence
    SOURCES tests/hle/test_render_cadence.cpp
    LIBRARIES prosper_core)

  add_executable(test_prosper_app_pad_overlay frontends/prosper-app/test_pad_overlay.cpp)
  target_include_directories(test_prosper_app_pad_overlay PRIVATE src frontends/prosper-app)
  target_link_libraries(test_prosper_app_pad_overlay PRIVATE prosper_core)
  add_test(NAME prosper_app_pad_overlay COMMAND test_prosper_app_pad_overlay)

  # WHAT the --fps counter says, kept out of the ImGui/Vulkan overlay that draws it. The order of the
  # two rates is the load-bearing part: a HUD that led with the PRESENTED rate would read ~60 fps for
  # a frozen title (#2783).
  add_executable(test_prosper_app_fps_hud frontends/prosper-app/test_fps_hud.cpp
                                          src/gpu/present/present_frame_rate.cpp)
  target_include_directories(test_prosper_app_fps_hud PRIVATE src frontends/prosper-app)
  add_test(NAME prosper_app_fps_hud COMMAND test_prosper_app_fps_hud)

  # Which key is which pad control (#2234). No SDL here: the frontend keeps the Key -> scancode
  # table and the mapping itself lives in a header, so this runs in the core suite on every host.
  add_executable(test_prosper_app_keyboard_pad_map frontends/prosper-app/test_keyboard_pad_map.cpp)
  target_include_directories(test_prosper_app_keyboard_pad_map PRIVATE src frontends/prosper-app)
  target_link_libraries(test_prosper_app_keyboard_pad_map PRIVATE prosper_core)
  add_test(NAME prosper_app_keyboard_pad_map COMMAND test_prosper_app_keyboard_pad_map)

  add_executable(test_prosper_app_window_controls frontends/prosper-app/test_window_controls.cpp)
  target_include_directories(test_prosper_app_window_controls PRIVATE frontends/prosper-app)
  add_test(NAME prosper_app_window_controls COMMAND test_prosper_app_window_controls)

  add_executable(test_prosper_app_snap_author frontends/prosper-app/test_snap_author.cpp)
  target_include_directories(test_prosper_app_snap_author PRIVATE frontends/prosper-app)
  add_test(NAME prosper_app_snap_author COMMAND test_prosper_app_snap_author)

  # Exit status + fault wording for a dead guest (#4140); no SDL/Vulkan.
  add_executable(test_prosper_app_guest_end_status frontends/prosper-app/test_guest_end_status.cpp)
  target_include_directories(test_prosper_app_guest_end_status PRIVATE frontends/prosper-app)
  add_test(NAME prosper_app_guest_end_status COMMAND test_prosper_app_guest_end_status)

  add_executable(test_prosper_app_capture_schedule frontends/prosper-app/test_capture_schedule.cpp)
  target_include_directories(test_prosper_app_capture_schedule PRIVATE frontends/prosper-app)
  add_test(NAME prosper_app_capture_schedule COMMAND test_prosper_app_capture_schedule)

  add_executable(test_prosper_app_performance_capture_schedule
    frontends/prosper-app/test_performance_capture_schedule.cpp)
  target_include_directories(test_prosper_app_performance_capture_schedule
    PRIVATE frontends/prosper-app)
  add_test(NAME prosper_app_performance_capture_schedule
    COMMAND test_prosper_app_performance_capture_schedule)

  add_executable(test_prosper_app_present_mode frontends/prosper-app/test_present_mode.cpp)
  target_include_directories(test_prosper_app_present_mode PRIVATE frontends/prosper-app)
  add_test(NAME prosper_app_present_mode COMMAND test_prosper_app_present_mode)

  # The interactive open path's path-resolution + action decisions (#1469): pure and SDL-free, so
  # the drag-and-drop / folder-picker behaviour is covered in normal CI without a window.
  add_executable(test_prosper_app_game_path frontends/prosper-app/test_game_path.cpp)
  target_include_directories(test_prosper_app_game_path PRIVATE frontends/prosper-app)
  add_test(NAME prosper_app_game_path COMMAND test_prosper_app_game_path)

  # The game library scan, param.json metadata, and settings precedence (#1471): filesystem access is
  # injected, so the whole scan runs in normal CI with no window, no ImGui and no real dump.
  add_executable(test_prosper_app_game_library frontends/prosper-app/test_game_library.cpp)
  target_include_directories(test_prosper_app_game_library PRIVATE frontends/prosper-app)
  add_test(NAME prosper_app_game_library COMMAND test_prosper_app_game_library)

  # F9 frame-grab file naming: one stamp and one exclusively claimed name pair per capture, so two
  # titles in one directory cannot overwrite each other and an aborted grab cannot pair with a
  # different capture. Uses a real directory under the build tree and an injected clock — no SDL, no
  # window, no GPU.
  add_executable(test_prosper_app_frame_grab_naming frontends/prosper-app/test_frame_grab_naming.cpp)
  target_include_directories(test_prosper_app_frame_grab_naming PRIVATE frontends/prosper-app)
  add_test(NAME prosper_app_frame_grab_naming COMMAND test_prosper_app_frame_grab_naming)

  add_executable(test_prosper_app_frame_grab_match frontends/prosper-app/test_frame_grab_match.cpp)
  target_include_directories(test_prosper_app_frame_grab_match PRIVATE frontends tests)
  target_compile_definitions(test_prosper_app_frame_grab_match PRIVATE
    PROSPER_APP_MAIN_SOURCE="${CMAKE_CURRENT_SOURCE_DIR}/frontends/prosper-app/main.cpp")
  add_test(NAME prosper_app_frame_grab_match COMMAND test_prosper_app_frame_grab_match)

  # F8's pre-trigger ring and post-trigger timing artifact: pure state machine + filesystem test,
  # including bounded overflow accounting and temporary-to-final atomic publication.
  prosper_add_gtest(test_texture_source_snapshot
    SOURCES frontends/shared/tests/test_texture_source_snapshot.cpp
    INCLUDES frontends)

  # Actual tracked mappings and direct comparison/window observations; no Vulkan device.
  add_executable(test_texture_direct_validation frontends/shared/tests/test_texture_direct_validation.cpp)
  target_include_directories(test_texture_direct_validation PRIVATE frontends)
  target_link_libraries(test_texture_direct_validation PRIVATE prosper_core)
  add_test(NAME texture_direct_validation COMMAND test_texture_direct_validation texture-direct-validation.jsonl)
  add_test(NAME texture_direct_validation_disabled COMMAND test_texture_direct_validation texture-direct-validation-disabled.jsonl)
  set_tests_properties(texture_direct_validation_disabled PROPERTIES
    ENVIRONMENT "PROSPER_NO_PERF_ALARMS=1")

  prosper_add_gtest(test_texture_validation_census
    SOURCES frontends/shared/tests/test_texture_validation_census.cpp
    INCLUDES frontends)

  add_executable(test_performance_capture frontends/shared/tests/test_performance_capture.cpp)
  target_link_libraries(test_performance_capture PRIVATE prosper_performance_capture)
  add_test(NAME performance_capture COMMAND test_performance_capture)

  if(Python3_Interpreter_FOUND)
    add_test(NAME performance_capture_report_logic
      COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/perf/test_performance_capture_report.py")
    add_test(NAME perf_f8_gap_logic
      COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/perf/test_perf_f8_gap.py")
    add_test(NAME schedstat_f8_logic
      COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/perf/test_schedstat_f8.py")
    add_test(NAME schedstat_probe_logic
      COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/perf/test_schedstat_probe.py")
    # RUN_SERIAL: its fixture changes are wall-clock timers that must land inside a fixed sampling
    # window, and under the CI's `ctest -j4` (#4116) they missed it on macOS and Windows in two
    # different subtests while passing every serial run. A stopgap, NOT a fix -- #4142.
    set_tests_properties(schedstat_probe_logic PROPERTIES RUN_SERIAL TRUE)
    add_test(NAME compare_capture_pair_logic
      COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/perf/test_compare_capture_pair.py")
  endif()

  # Library grid selection movement (#1471): arithmetic only, so the navigation rules are covered in
  # normal CI rather than by someone pressing arrow keys. No ImGui, no SDL, no window.
  add_executable(test_prosper_app_library_nav frontends/prosper-app/test_library_nav.cpp)
  target_include_directories(test_prosper_app_library_nav PRIVATE frontends/prosper-app)
  add_test(NAME prosper_app_library_nav COMMAND test_prosper_app_library_nav)

  # Library descriptor-pool sizing and the exhaustion rule (#1649): the pool must follow the title count
  # instead of a fixed 256, and a set must never be allocated out of a full pool — ImGui's backend writes
  # to whatever vkAllocateDescriptorSets returned without checking it. Pure arithmetic plus a fake
  # allocator that records whether it ran, so no Vulkan, no ImGui and no window.
  add_executable(test_prosper_app_library_descriptor_budget
                 frontends/prosper-app/test_library_descriptor_budget.cpp)
  target_include_directories(test_prosper_app_library_descriptor_budget PRIVATE frontends/prosper-app)
  add_test(NAME prosper_app_library_descriptor_budget
           COMMAND test_prosper_app_library_descriptor_budget)

  # Library background art and focus music (#1630): DDS/BC7 and RIFF/ATRAC9 container validation plus
  # the debounce, two-stage fade and LRU policy. Pure byte and arithmetic work — the containers are
  # synthesized in the test, so no dump, no GPU and no audio device is involved.
  add_executable(test_prosper_app_launcher_media frontends/prosper-app/test_launcher_media.cpp)
  target_include_directories(test_prosper_app_launcher_media PRIVATE frontends/prosper-app)
  add_test(NAME prosper_app_launcher_media COMMAND test_prosper_app_launcher_media)

  # Fault-safe guest-pointer read in the PROSPER_AMPRLOG diagnostic (#1154): an unmapped, aligned,
  # > 0xffff pointer (GTA V's 0x302200) must report unreadable, not SIGSEGV. POSIX-only
  # (process_vm_readv / mmap); ampr_arglog lives in the __linux__||__APPLE__ half of hle_kernel_mem.
  if(NOT WIN32)
    # Does not link prosper_core, so it does not inherit its PUBLIC `src`.
    prosper_add_gtest(test_amprlog_safe_read
      SOURCES tests/host/platform/test_amprlog_safe_read.cpp INCLUDES src)
  endif()

  # Compute-side renderer-owned-RTT scale detection (PROSPER_RENDER_SCALE>1 downscale vs a view alias).
  # Pure integer arithmetic, no deps.
  prosper_add_gtest(test_rtt_scale
    SOURCES frontends/shared/tests/test_rtt_scale.cpp
    INCLUDES frontends)

  # MRT-prefix truncation: which bound colour slot may join a pass, and which extent comparisons are
  # evidence of a real conflict rather than an unmeasured 0x0. Pure integer arithmetic, no deps.
  prosper_add_gtest(test_mrt_extent
    SOURCES frontends/shared/tests/test_mrt_extent.cpp
    INCLUDES frontends)

  # ONE active-colour-binding policy, shared by pass grouping and same-pass feedback detection, built
  # from a real DrawItem. A second, looser copy of this rule classified stale named state as a live
  # binding (#2550 review).
  prosper_add_gtest(test_mrt_binding
    SOURCES frontends/shared/tests/test_mrt_binding.cpp
    INCLUDES frontends src)

  prosper_add_gtest(test_rtt_injection
    SOURCES frontends/shared/tests/test_rtt_injection.cpp
    INCLUDES frontends)

  prosper_add_gtest(test_volume_target_index
    SOURCES frontends/shared/tests/test_volume_target_index.cpp
    INCLUDES frontends)

  prosper_add_gtest(test_buffer_range_plan
    SOURCES frontends/shared/tests/test_buffer_range_plan.cpp
    INCLUDES tests)

  prosper_add_gtest(test_readback_policy
    SOURCES frontends/shared/tests/test_readback_policy.cpp
    INCLUDES frontends)

  prosper_add_gtest(test_resolve_submission_policy
    SOURCES frontends/shared/tests/test_resolve_submission_policy.cpp
    INCLUDES frontends)

  prosper_add_gtest(test_capture_renderer_policy
    SOURCES frontends/shared/tests/test_capture_renderer_policy.cpp
    INCLUDES frontends)

  # Which sampled resource is AvPlayer's NV12 chroma plane (#2005). Descriptor-level classification
  # only — no device — but it reads the real guest-linear pitch registry, so it links prosper_core.
  prosper_add_gtest(test_avplayer_plane_policy
    SOURCES frontends/shared/tests/test_avplayer_plane_policy.cpp
    INCLUDES frontends src
    LIBRARIES prosper_core)

  # The publish extent contract (#1986): which rendered pass may become the present source, and what
  # the retained-frame net may retain. Pure byte-count policy, so no device and no renderer.
  prosper_add_gtest(test_present_extent
    SOURCES frontends/shared/tests/test_present_extent.cpp
    INCLUDES frontends)

  # When prosper may publish the guest's OWN flipped display buffer (#1968), including the guest
  # authorship precondition whose absence reverted the first attempt (#2044). Pure policy, no device.
  prosper_add_gtest(test_guest_scanout_present
    SOURCES frontends/shared/tests/test_guest_scanout_present.cpp
    INCLUDES frontends)

  # The census window PROSPER_PASS_LOG / PROSPER_DUMP_PERSISTENT open: an ordinal (unchanged) or
  # `ms:<millis>` (#1968). Pure predicate — no renderer, no device, and the clock is an argument.
  prosper_add_gtest(test_diagnostic_window
    SOURCES frontends/shared/tests/test_diagnostic_window.cpp
    INCLUDES frontends)

  # The optional retained-target selector bounds readback without changing the legacy broad
  # census. Malformed and oversized lists must decline every target rather than read them all.
  prosper_add_gtest(test_persistent_readback_filter
    SOURCES frontends/shared/tests/test_persistent_readback_filter.cpp
    INCLUDES frontends src
    LIBRARIES prosper_core)

  # boot_trace periodic BMPs are opt-in, with the legacy disable taking priority. This exercises the
  # exact environment-name lookup and registrar forwarding without constructing Vulkan or writing files.
  add_executable(test_frame_dump_policy frontends/shared/tests/test_frame_dump_policy.cpp)
  target_include_directories(test_frame_dump_policy PRIVATE frontends)
  add_test(NAME frame_dump_policy COMMAND test_frame_dump_policy)

  prosper_add_gtest(test_capture_compute_policy
    SOURCES tests/gpu/capture/test_capture_compute_policy.cpp
    INCLUDES src)

  add_executable(test_write_watch_policy frontends/shared/tests/test_write_watch_policy.cpp)
  # `src` as well as `frontends`: the knobs' strict parsing lives in diagnostics/env_numeric.hpp,
  # and it is asserted here beside the policy whose sentinels make a typo dangerous (#3253).
  target_include_directories(test_write_watch_policy PRIVATE frontends src)
  add_test(NAME write_watch_policy COMMAND test_write_watch_policy)

  # Pooled texture-decode intermediates. `src` for diagnostics/env_numeric.hpp, which the pool's
  # retention knob parses through -- the equivalence arm is the one that matters (see the file head).
  prosper_add_gtest(test_decode_scratch
    SOURCES frontends/shared/tests/test_decode_scratch.cpp
    INCLUDES frontends src)
  # The two arms the pooling claim rests on, each asserting the COMPLEMENT of the default run so
  # neither can pass by accident: with retention disabled the shared pool must keep nothing (and a
  # ScratchBuffer must therefore not reuse pages), and with poison armed every lease must arrive
  # holding 0xCD rather than the incidental zeros a fresh mmap would have given it.
  add_test(NAME decode_scratch_no_retention COMMAND test_decode_scratch)
  set_tests_properties(decode_scratch_no_retention PROPERTIES
    ENVIRONMENT "PROSPER_DECODE_SCRATCH_MB=0")
  add_test(NAME decode_scratch_poisoned COMMAND test_decode_scratch)
  set_tests_properties(decode_scratch_poisoned PROPERTIES
    ENVIRONMENT "PROSPER_DECODE_SCRATCH_POISON=1")

  # The pool's reuse census reports at end of run through diagnostics/exit_reports, so this
  # target needs `src` on its include path and that translation unit linked in.
  add_executable(test_cpu_rtt_snapshot_pool
    frontends/shared/tests/test_cpu_rtt_snapshot_pool.cpp
    src/diagnostics/exit_reports.cpp)
  target_include_directories(test_cpu_rtt_snapshot_pool PRIVATE frontends src)
  target_sources(test_cpu_rtt_snapshot_pool PRIVATE src/diagnostics/worker_spawn_census.cpp
                                                    src/diagnostics/exit_census.cpp
                                                    src/diagnostics/transfer_pressure.cpp)
  target_link_libraries(test_cpu_rtt_snapshot_pool PRIVATE Threads::Threads)
  add_test(NAME cpu_rtt_snapshot_pool COMMAND test_cpu_rtt_snapshot_pool)

  add_executable(test_depth_cube_quantize frontends/shared/tests/test_depth_cube_quantize.cpp)
  target_include_directories(test_depth_cube_quantize PRIVATE frontends)
  add_test(NAME depth_cube_quantize COMMAND test_depth_cube_quantize)
  add_test(NAME depth_cube_quantize_scalar COMMAND test_depth_cube_quantize --control)
  set_tests_properties(depth_cube_quantize_scalar PROPERTIES
    ENVIRONMENT "PROSPER_NO_DEPTH_CUBE_SIMD=1")

  # #3307 compute->graphics image borrow census. Header-only and device-free: the load-bearing arm
  # is the 2048-case equivalence between the importer's precondition classifier and the `||` chain
  # it replaced, which needs no Vulkan and must never be skipped on a host without it.
  prosper_add_gtest(test_compute_image_borrow_census
    SOURCES frontends/shared/tests/test_compute_image_borrow_census.cpp
    INCLUDES frontends)

  # Storage-image alias access union is metadata-only and must run without Vulkan on every platform.
  prosper_add_gtest(test_storage_image_alias_plan
    SOURCES frontends/shared/tests/test_storage_image_alias_plan.cpp
    INCLUDES frontends src
    LIBRARIES prosper_core)

  prosper_add_gtest(test_texture_decode_diagnostic
    SOURCES frontends/shared/tests/test_texture_decode_diagnostic.cpp
    INCLUDES frontends src
    LIBRARIES prosper_core)

  # Bounded-acquire present classification (#1182). Needs Vulkan HEADERS only (VkResult enum), no loader
  # call — link Vulkan::Vulkan purely for its include dirs. Guarded so a headers-less host skips it
  # gracefully; every platform that builds prosper-app has Vulkan available.
  if(NOT TARGET Vulkan::Vulkan)
    find_package(Vulkan QUIET)
  endif()
  if(TARGET Vulkan::Vulkan)
    add_executable(test_prosper_app_present_policy frontends/prosper-app/test_present_policy.cpp)
    target_include_directories(test_prosper_app_present_policy PRIVATE frontends/prosper-app)
    target_link_libraries(test_prosper_app_present_policy PRIVATE Vulkan::Vulkan)
    add_test(NAME prosper_app_present_policy COMMAND test_prosper_app_present_policy)

    prosper_add_gtest(test_present_blit_policy
      SOURCES frontends/shared/tests/test_present_blit_policy.cpp
      INCLUDES frontends src
      LIBRARIES Vulkan::Vulkan)
  endif()

  prosper_add_gtest(test_sync_on_address
    SOURCES tests/hle/test_sync_on_address.cpp
    LIBRARIES prosper_core)

  # PROSPER_SYNC_RING is sized from the environment at static init, so the ring size must be set
  # for the process rather than by the test body. 64 is deliberately tiny: the property under test is
  # what the dump reports once it has DROPPED events.
  add_executable(test_sync_ring tests/hle/sync/test_sync_ring.cpp)
  target_link_libraries(test_sync_ring PRIVATE prosper_core)
  add_test(NAME sync_ring COMMAND test_sync_ring)
  set_tests_properties(sync_ring PROPERTIES ENVIRONMENT "PROSPER_SYNC_RING=64")

  # PROSPER_SYNC_RING_CALLERS: ring events carry the guest call site that reached them, so a total
  # deadlock can be traced back to the guest function that was supposed to signal.
  prosper_add_gtest(test_sync_ring_callers
    SOURCES tests/hle/sync/test_sync_ring_callers.cpp
    INCLUDES src
    LIBRARIES prosper_core
    ENVIRONMENT PROSPER_SYNC_RING=256 PROSPER_SYNC_RING_CALLERS=1)

  prosper_add_gtest(test_mutex_host_tcb_identity
    SOURCES tests/hle/sync/test_mutex_host_tcb_identity.cpp
    LIBRARIES prosper_core)

  prosper_add_gtest(test_fiber_guest_tp_stash
    SOURCES tests/hle/sync/test_fiber_guest_tp_stash.cpp
    LIBRARIES prosper_core)

  prosper_add_gtest(test_cond_slot_identity
    SOURCES tests/hle/sync/test_cond_slot_identity.cpp
    LIBRARIES prosper_core)

  # sceKernelDlsym honors its module handle (#147): per-module export tables, real handles from
  # LoadStartModule for linked-module paths, global first-wins fallback. Pure, no game dump needed.
  prosper_add_gtest(test_dlsym_handle
    SOURCES tests/hle/test_dlsym_handle.cpp
    LIBRARIES prosper_core)

  # Guest mutex TYPE semantics (#145): settype honored, explicit Sony/FreeBSD ERRORCHECK default,
  # non-recursive static-sentinel self-init, FreeBSD errno values (EDEADLK=11). Pure, no game dump.
  prosper_add_gtest(test_mutex_types
    SOURCES tests/hle/kernel/test_mutex_types.cpp
    LIBRARIES prosper_core)

  # pthread key destructor teardown at thread exit through the scePthread* HLE,
  # including the Windows (winpthreads) path: single fire, reinstall
  # refire/stop, iteration bound, deleted-key silence, all keys fire with no
  # order asserted, NULL destructor silence. Pure, no game dump.
  prosper_add_gtest(test_pthread_key_teardown
    SOURCES tests/hle/kernel/test_pthread_key_teardown.cpp
    LIBRARIES prosper_core
    PROPERTIES TIMEOUT 60)

  prosper_add_gtest(test_agc_shader
    SOURCES tests/gpu/test_agc_shader.cpp
    LIBRARIES prosper_core)

  prosper_add_gtest(test_tile
    SOURCES tests/gpu/texture/test_tile.cpp
    LIBRARIES prosper_core)
  add_test(NAME gpu_surface_detile_gather COMMAND test_tile)
  set_tests_properties(gpu_surface_detile_gather PROPERTIES
                       ENVIRONMENT "PROSPER_NO_PAIRED_AVX2_DETILE=1")
  # Explicit CPU microbenchmark for the recurring 4K, eight-byte Astro storage seed.
  add_executable(bench_detile tests/gpu/texture/bench_detile.cpp)
  target_link_libraries(bench_detile PRIVATE prosper_core)
  add_executable(test_mip_tail tests/gpu/texture/test_mip_tail.cpp)
  target_link_libraries(test_mip_tail PRIVATE prosper_core)
  add_test(NAME gpu_mip_tail_oracle COMMAND test_mip_tail --check)
  set_tests_properties(gpu_mip_tail_oracle PROPERTIES TIMEOUT 60)
  if(Python3_Interpreter_FOUND)
    foreach(pipes 1 2 4 8 16 32 64)
      add_test(NAME gpu_mip_tail_pipes_${pipes} COMMAND ${CMAKE_COMMAND} -E env
        PROSPER_RX_PIPES=${pipes} ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/gpu/texture/test_mip_tail.py $<TARGET_FILE:test_mip_tail>)
      set_tests_properties(gpu_mip_tail_pipes_${pipes} PROPERTIES TIMEOUT 150)
    endforeach()
  endif()
  # Run the same byte-identity suite through the row-parallel detile path (#1177), forcing threads so
  # it is exercised even on a low-core CI. Passing here + in gpu_surface_detile (single-threaded control
  # via the < 2-thread gate / PROSPER_DETILE_SINGLE_THREADED) proves threaded == single byte-for-byte.
  add_test(NAME gpu_surface_detile_threaded COMMAND test_tile)
  set_tests_properties(gpu_surface_detile_threaded PROPERTIES
                       ENVIRONMENT "PROSPER_DETILE_THREADS=8")

  # BC1/BC2/BC3 block decompression to RGBA8 (#121): hand-built blocks with known output. Pure.
  add_executable(test_bc_decode tests/gpu/texture/test_bc_decode.cpp)
  target_link_libraries(test_bc_decode PRIVATE prosper_core)
  add_test(NAME bc_decode COMMAND test_bc_decode)

  prosper_add_gtest(test_stat
    SOURCES tests/test_stat.cpp
    LIBRARIES prosper_core)

  # Includes a shared fixture, which resolves from the `tests` root.
  prosper_add_gtest(test_file_binary
    SOURCES tests/hle/test_file_binary.cpp
    INCLUDES tests
    LIBRARIES prosper_core)

  # The guest's fopen and stdio stream handlers, against N1570 7.21.5.3 and the guest libc's own
  # Dinkumware `_Foprep` mode loop (guest_fopen_mode.hpp):
  # guest-path translation, mode semantics, EINVAL for a bad mode. Scratch files only, no dump.
  prosper_add_gtest(test_hle_fopen_modes
    SOURCES tests/hle/test_hle_fopen_modes.cpp
    INCLUDES tests
    LIBRARIES prosper_core)
  if(UNIX)
    # Intercepts both the old host-wide and new filesystem-wide barrier; no real writeback.
    add_executable(test_file_sync tests/hle/test_file_sync.cpp)
    target_include_directories(test_file_sync PRIVATE tests)
    target_link_libraries(test_file_sync PRIVATE prosper_core)
    add_test(NAME file_sync_scope COMMAND test_file_sync)
    set_tests_properties(file_sync_scope PROPERTIES TIMEOUT 20)
  endif()

  prosper_add_gtest(test_savedata_dirname
    SOURCES tests/hle/test_savedata_dirname.cpp
    LIBRARIES prosper_core)

  prosper_add_gtest(test_savedata_txres
    SOURCES tests/hle/test_savedata_txres.cpp
    LIBRARIES prosper_core)

  prosper_add_gtest(test_savedata_event_drain
    SOURCES tests/hle/service/test_savedata_event_drain.cpp
    INCLUDES tests
    LIBRARIES prosper_core)

  # libScePsml must report failure rather than the dispatcher's default SCE_OK: the guest reads an
  # out-struct prosper never wrote and allocates from it (PPSA20447). Pure, no dump, no device.
  prosper_add_gtest(test_psml_unimplemented
    SOURCES tests/hle/service/test_psml_unimplemented.cpp
    LIBRARIES prosper_core)

  # libSceSigninDialog runs its common-dialog lifecycle and ends dismissed (#3784). Pure, no dump.
  prosper_add_gtest(test_signin_dialog
    SOURCES tests/hle/service/test_signin_dialog.cpp
    LIBRARIES prosper_core)

  # Includes a shared fixture, which resolves from the `tests` root.
  prosper_add_gtest(test_translate_sandbox
    SOURCES tests/hle/test_translate_sandbox.cpp
    INCLUDES tests
    LIBRARIES prosper_core)

  add_executable(test_service_logging tests/hle/test_service_logging.cpp)
  target_link_libraries(test_service_logging PRIVATE prosper_core)
  add_test(NAME service_logging_unmapped_args COMMAND test_service_logging)

  if(WIN32)
    add_executable(test_win_exception_delivery tests/misc/test_win_exception_delivery.cpp)
    prosper_add_gtest(test_native_host_wait_producers
      SOURCES tests/hle/kernel/test_native_host_wait_producers.cpp
      INCLUDES tests
      LIBRARIES prosper_core
      ENVIRONMENT PROSPER_HOST_WAIT_OBSERVE=1
      PROPERTIES TIMEOUT 15 ENVIRONMENT_MODIFICATION "PROSPER_WAITCAP=unset:")
    # Includes a shared fixture, which resolves from the `tests` root.
    target_include_directories(test_win_exception_delivery PRIVATE tests)
    target_link_libraries(test_win_exception_delivery PRIVATE prosper_core)
    add_test(NAME win_exception_delivery COMMAND test_win_exception_delivery)

    add_executable(test_win_pthread_key_destructor tests/hle/test_win_pthread_key_destructor.cpp)
    target_link_libraries(test_win_pthread_key_destructor PRIVATE prosper_core)
    add_test(NAME win_pthread_key_destructor COMMAND test_win_pthread_key_destructor)

    add_executable(test_win_kernel_is_stack tests/host/image/test_win_kernel_is_stack.cpp)
    target_link_libraries(test_win_kernel_is_stack PRIVATE prosper_core)
    add_test(NAME win_kernel_is_stack COMMAND test_win_kernel_is_stack)

    # PROSPER_HWWATCH_ABS: the Windows data write-watchpoint (debug registers, per guest thread).
    prosper_add_gtest(test_win_data_watch
      SOURCES tests/host/image/test_win_data_watch.cpp
      LIBRARIES prosper_core)
  endif()

  # Boot preload path case correction (#1006): a module present with different casing than the
  # hard-coded preload path (Blasphemous 2 / Evergate's Il2CppUserAssemblies.prx on a case-sensitive
  # host filesystem) must resolve to the real on-disk entry instead of being dropped as absent.
  # Pure std::filesystem against a temp tree; no game dump needed.
  # Includes a shared fixture, which resolves from the `tests` root.
  prosper_add_gtest(test_module_path_case
    SOURCES tests/host/image/test_module_path_case.cpp
    INCLUDES tests
    LIBRARIES prosper_core)

  # Module path policy: prosper links guest modules only from the dump locations it names
  # (eboot.bin, sce_module/, Media/Modules/, Media/Plugins/), rejects everything else by default,
  # and never links a Sony-named library out of the wholesale-auto-linked Media directories. Some
  # dumps ship third-party replacements of libSceAppContent / libSceNpEntitlementAccess /
  # libSceAmpr; loading one would shadow prosper's own implementation and answer the guest with
  # somebody else's stub. Pure string policy plus one end-to-end arm against a temp tree.
  prosper_add_gtest(test_module_path_policy
    SOURCES tests/host/image/test_module_path_policy.cpp
    INCLUDES tests
    LIBRARIES prosper_core)

  # dump_hygiene classifies files in a game dump that prosper must never load, and can delete them.
  # Its control arms matter more than its positive ones: during development it twice classified
  # AUTHENTIC dump content as removable. The suite pins both regressions and the guards behind them.
  add_test(NAME dump_hygiene_classifier
           COMMAND ${Python3_EXECUTABLE} -m unittest tests.tools.test_dump_hygiene
           WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

  # End-of-run reports reachable on the exit paths runs actually take (#3353). Its subprocess arms
  # re-run the test binary: one exits with _Exit and no flush as the positive control that atexit
  # really is skipped, one flushes first, one returns from main to cover the atexit fallback.
  add_executable(test_exit_reports tests/diagnostics/test_exit_reports.cpp)
  target_link_libraries(test_exit_reports PRIVATE prosper_core)
  add_test(NAME exit_reports COMMAND test_exit_reports)

  # [perf-alarm] rules and engine (#3891). Every rule gets a window built by hand from the numbers
  # measured before the fix it was written against (fires) and after it (quiet), plus a boundary
  # arm; the engine arms pin window closing, per-rule log de-duplication, JSONL and the summary.
  prosper_add_gtest(test_perf_alarms
    SOURCES tests/diagnostics/test_perf_alarms.cpp
    LIBRARIES prosper_core)
  # The run-level frame-breakdown line of the exit summary: per-flip means, the sampled cost it must
  # exclude, the "not a partition" wording, and accumulation through the engine. Its own executable so
  # it needs none of test_perf_alarms' HLE-dispatcher fixtures.
  prosper_add_gtest(test_frame_breakdown
    SOURCES tests/diagnostics/test_frame_breakdown.cpp
    LIBRARIES prosper_core)
  # The rules' detail strings go through a printf-attributed format() helper; the attribute only
  # checks anything under -Wformat, which prosper_core does not enable, so make a mismatch an ERROR
  # for this one translation unit (the gpu_replay precedent below: GCC diagnoses this at plain
  # -Wformat). Positive control: "%s" given an unsigned long long fails the build.
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set_source_files_properties(src/diagnostics/perf/perf_alarm_rules.cpp
                                PROPERTIES COMPILE_OPTIONS "-Werror=format")
  endif()

  # Diagnostics: disabled-path test — verifies stubs are zero-cost (always runs).
  prosper_add_gtest(test_diagnostics_disabled
    SOURCES tests/diagnostics/test_diagnostics_disabled.cpp
    LIBRARIES prosper_core)

  # The boot-phase LINE is reachable in a default build (no PROSPER_DIAGNOSTICS), driven only by
  # PROSPER_BOOTPHASE. Deliberately registered outside the PROSPER_DIAGNOSTICS block: the property
  # under test is that the default build — the one everybody actually runs — can be made to say
  # where boot_program() stopped.
  # Bundled support PRX policy and actual candidate discovery over synthetic module files.
  prosper_add_gtest(test_module_entry_abi
    SOURCES tests/host/image/test_module_entry_abi.cpp
    INCLUDES src
    LIBRARIES prosper_core)
  prosper_add_gtest(test_support_modules
    SOURCES tests/loader/test_support_modules.cpp
    INCLUDES src tests
    LIBRARIES prosper_core
    PROPERTIES ENVIRONMENT_MODIFICATION
      "PROSPER_NO_PLUGIN_AUTOLINK=unset:")

  # Which preloaded modules are started with a real SCE module-param descriptor. Pure policy over
  # the fixed guest base map: no dump, no SELF parser, no filesystem — so both the inclusion and the
  # exclusion direction are checkable in ordinary CI.
  prosper_add_gtest(test_module_start_params
    SOURCES tests/host/image/test_module_start_params.cpp src/host/image/module_start_params.cpp
    INCLUDES src)

  add_executable(test_boot_phase_log tests/diagnostics/test_boot_phase_log.cpp)
  target_link_libraries(test_boot_phase_log PRIVATE prosper_core)
  add_test(NAME diagnostics_boot_phase_log COMMAND test_boot_phase_log)

  # PROSPER_ENV_ON / PROSPER_ENV_VALUE sample getenv exactly once (#3094). Header-only, so this
  # needs no library: it mutates the environment between two evaluations of the same macro and
  # asserts the second still reports the first value -- the one behaviour a live getenv cannot
  # produce, so the arms go red if the caching is ever removed.
  prosper_add_gtest(test_env_cache
    SOURCES tests/diagnostics/test_env_cache.cpp
    INCLUDES src)

  prosper_add_gtest(test_native_host_wait
    SOURCES tests/diagnostics/test_native_host_wait.cpp
    LIBRARIES prosper_core
    PROPERTIES TIMEOUT 15)
  prosper_add_gtest(test_native_host_wait_disabled
    SOURCES tests/diagnostics/test_native_host_wait_disabled.cpp
    LIBRARIES prosper_core
    PROPERTIES TIMEOUT 15 ENVIRONMENT_MODIFICATION "PROSPER_HOST_WAIT_OBSERVE=unset:")

  # The per-submit sampling contract (diagnostics/env_submit.hpp). Two-sided by construction: it
  # asserts a value is frozen INSIDE one SubmitEnvScope -- which a live getenv cannot do -- and that
  # the next scope observes an arm made since -- which PROSPER_ENV_ON cannot do. Either half alone
  # would pass against the wrong mechanism. Threads, so link the thread library.
  prosper_add_gtest(test_env_submit
    SOURCES tests/diagnostics/test_env_submit.cpp
    INCLUDES src
    LIBRARIES Threads::Threads)

  # One arm per PROSPER_* knob #3267 converted to diagnostics/env_numeric.hpp. Header-only and
  # environment-free by construction: nearly every converted site caches its read in a
  # function-local static, so an arm that armed the variable at runtime would go VACUOUS rather
  # than red (#2214). Each arm therefore calls the same helper the site calls, and also computes
  # what the site's OLD spelling answered -- so an arm on a knob that never needed converting
  # fails instead of passing quietly.
  add_executable(test_env_numeric_sites tests/diagnostics/test_env_numeric_sites.cpp)
  target_include_directories(test_env_numeric_sites PRIVATE src)
  add_test(NAME diagnostics_env_numeric_sites COMMAND test_env_numeric_sites)

  # Diagnostics: enabled-path and JSON tests (only when feature is built).
  if(PROSPER_DIAGNOSTICS)
    add_executable(test_diagnostics_enabled tests/diagnostics/test_diagnostics_enabled.cpp)
    target_link_libraries(test_diagnostics_enabled PRIVATE prosper_core)
    target_compile_definitions(test_diagnostics_enabled PRIVATE PROSPER_DIAGNOSTICS)
    add_test(NAME diagnostics_enabled_capture COMMAND test_diagnostics_enabled)

    add_executable(test_diagnostics_json tests/diagnostics/test_diagnostics_json.cpp)
    target_link_libraries(test_diagnostics_json PRIVATE prosper_core)
    target_compile_definitions(test_diagnostics_json PRIVATE PROSPER_DIAGNOSTICS)
    add_test(NAME diagnostics_json_format COMMAND test_diagnostics_json)
  endif()

  # Unity native-plugin auto-link (#1609): a title's own Media/Plugins/*.prx that the fixed preload
  # list never named must still be linked, or its first P/Invoke raises a DllNotFoundException that
  # IL2CPP captures into a Task and never prints. Also guards the export-collision rule that keeps
  # auto-linking from introducing a duplicate NID export (Evergate ships libfmod.prx AND libfmodL.prx,
  # the same library's release and logging builds). The discovery and predicate assertions are pure
  # std::filesystem/struct work and need no dump; the dump, when configured, additionally exercises
  # link_program's honouring of the skip flag.
  add_executable(test_plugin_autolink tests/misc/test_plugin_autolink.cpp)
  # Includes a shared fixture, which resolves from the `tests` root.
  target_include_directories(test_plugin_autolink PRIVATE tests)
  target_link_libraries(test_plugin_autolink PRIVATE prosper_core)
  if(HAVE_GAME_DUMP)
    add_test(NAME plugin_autolink COMMAND test_plugin_autolink "${GAME_DUMP}")
    set_tests_properties(plugin_autolink PROPERTIES SKIP_RETURN_CODE 77)
  else()
    add_test(NAME plugin_autolink COMMAND test_plugin_autolink)
    set_tests_properties(plugin_autolink PROPERTIES SKIP_RETURN_CODE 77)
  endif()

  # link_program's structural contract on SYNTHESIZED modules (#2567). plugin_autolink's remaining
  # 14 assertions — the export table, the collision guard's effect on link_program, and the #1635
  # alias record — sat behind `if (argc >= 2)` plus a real Media/Plugins/PSN.prx, so they never ran
  # in CI and never ran for an agent without that title installed. These two cases register
  # UNCONDITIONALLY: their fixtures are minimal ET_SCE_DYNAMIC PRX modules emitted at test time by
  # tests/fixtures/synth_prx.h and tests/fixtures/handmade_prx.h, so there is no dump, no network and no committed
  # binary. Every byte is synthesized from the published ELF64 layout — nothing is carved out of a
  # dump, which would put Sony-built code in a public repository.
  #
  # Two cases, not one, because "the loader accepts this" and "the loader refuses that" are different
  # claims and the second is the one that gives the first its meaning: _reject holds the corrupted
  # variants, each differing from a linking control in exactly one field.
  # Includes a shared fixture, which resolves from the `tests` root.
  prosper_add_gtest(test_loader_synth_link
    SOURCES tests/misc/test_loader_synth_link.cpp
    INCLUDES tests
    LIBRARIES prosper_core)

  # Includes a shared fixture, which resolves from the `tests` root.
  prosper_add_gtest(test_loader_synth_reject
    SOURCES tests/misc/test_loader_synth_reject.cpp
    INCLUDES tests
    LIBRARIES prosper_core)

  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    # Fail sizing/seek operations on a valid fixture without blocking I/O or huge allocations.
    add_executable(test_module_file_io tests/misc/test_module_file_io.cpp)
    target_include_directories(test_module_file_io PRIVATE tests)
    target_link_libraries(test_module_file_io PRIVATE prosper_core)
    target_link_options(test_module_file_io PRIVATE
      "-Wl,--wrap=fseek" "-Wl,--wrap=ftell" "-Wl,--wrap=fread" "-Wl,--wrap=fclose")
    add_test(NAME module_file_io COMMAND test_module_file_io)
    set_tests_properties(module_file_io PROPERTIES TIMEOUT 20)
  endif()

  # ATRAC9 decode (vendored LibAtrac9 + prosper glue): an embedded 16-superframe slice of a real PS5
  # snd0.at9 decodes bit-exact vs the LibAtrac9 reference (CRC32). Pure, no game dump; runs in CI.
  # Includes a shared fixture, which resolves from the `tests` root.
  prosper_add_gtest(test_atrac9
    SOURCES tests/hle/audio/test_atrac9.cpp
    INCLUDES tests
    LIBRARIES prosper_core)

  # AJM batch-2.0 gapless decode semantics (#1097): skip/total trim, exact EOS landing of
  # uiTotalDecodedSamples, post-EOS silence, and program re-arm. FMOD's codec recycling keys off
  # the exact EOS; without it one-shot SFX exhaust the codec pool. Pure, no game dump; runs in CI.
  # Includes a shared fixture, which resolves from the `tests` root.
  prosper_add_gtest(test_ajm2_gapless
    SOURCES tests/hle/audio/test_ajm2_gapless.cpp
    INCLUDES tests
    LIBRARIES prosper_core)

  # APR container registry + read-source resolution (#62/#1901): stable ids, path dedup, ID-first
  # reads, unique-size fallback, ambiguous fallback refusal, and method diagnostics. Pure; no dump.
  # Includes a shared fixture, which resolves from the `tests` root.
  prosper_add_gtest(test_apr_registry
    SOURCES tests/host/memory/test_apr_registry.cpp
    INCLUDES tests
    LIBRARIES prosper_core)

  # #2384: the AMPR tracks-offset predicate. Its NEW case is provable by a boot; what needs a unit
  # test is that the widening is strictly ADDITIVE, because the predicate is shared by every title
  # that constructs a command buffer and none of their dumps are available to CI. Pure function of a
  # 64-bit argument, so it needs no dump and cannot skip silently.
  prosper_add_gtest(test_ampr_tracks_offset
    SOURCES tests/hle/memory/test_ampr_tracks_offset.cpp
    LIBRARIES prosper_core)
  prosper_add_gtest(test_ampr_amm
    SOURCES tests/hle/memory/test_ampr_amm.cpp
    LIBRARIES prosper_core)

  prosper_add_gtest(test_mprotect_unaligned
    SOURCES tests/hle/memory/test_mprotect_unaligned.cpp
    LIBRARIES prosper_core)

  # #2908: a refused Ampr map must hand its direct-memory claim back. POSIX-only because the
  # mapping primitives it exercises are — on Windows sceAmprCommandBufferSetBuffer is still the
  # do-nothing stub, which cannot leak and cannot demonstrate the lever either.
  if(NOT WIN32)
    prosper_add_gtest(test_ampr_map_refusal_releases_dmem
      SOURCES tests/hle/memory/test_ampr_map_refusal_releases_dmem.cpp
      LIBRARIES prosper_core)
  endif()

  # GTA V (#1139/#1142) net-new HLE regression guards: sceKernelAprGetFileStat fills the real stat
  # (#1133) and sceAudioOut2GetSpeakerArrayMemorySize returns a non-zero allocation size (#1134). The
  # AGC GetSize third piece is covered separately by test_agc_getsize (#1208).
  # Includes a shared fixture, which resolves from the `tests` root.
  prosper_add_gtest(test_apr_stat_audio
    SOURCES tests/hle/test_apr_stat_audio.cpp
    INCLUDES tests
    LIBRARIES prosper_core)

  # #2926/#2928: the APR read builders' two silent-success shapes -- an unregistered NID answering
  # SCE_OK from the dispatcher's return-0 default, and a gather/scatter segment reporting success
  # after its bytes failed to reach the guest destination -- plus the chain closer set, which the
  # handler's comment used to under-state. Pure: a scratch file and generated import stubs, no dump.
  # Includes a shared fixture, which resolves from the `tests` root.
  prosper_add_gtest(test_apr_gather_scatter
    SOURCES tests/hle/test_apr_gather_scatter.cpp
    INCLUDES tests
    LIBRARIES prosper_core)
  # #3245: sceAmprMeasureCommandSizeReadFile's compatibility read fired for every title, so a title
  # that also submits its measured reads paid for the same bytes twice (13,925 duplicate reads /
  # 582 MB in a 3-minute Stray boot) and had them written before it recorded the command. Registered
  # twice on purpose -- the PROSPER_APR_MEASURE_READ=always run asserts the opposite outcome, which
  # is what shows the suppression actually suppressed. Pure: scratch files and generated import
  # stubs, no game dump.
  # Includes a shared fixture, which resolves from the `tests` root.
  prosper_add_gtest(test_apr_measure_duplicate
    SOURCES tests/hle/test_apr_measure_duplicate.cpp
    INCLUDES tests
    LIBRARIES prosper_core)
  add_test(NAME apr_measure_duplicate_forced COMMAND test_apr_measure_duplicate)
  set_tests_properties(apr_measure_duplicate_forced PROPERTIES
                       ENVIRONMENT "PROSPER_APR_MEASURE_READ=always")
  add_test(NAME apr_measure_duplicate_never COMMAND test_apr_measure_duplicate)
  set_tests_properties(apr_measure_duplicate_never PROPERTIES
                       ENVIRONMENT "PROSPER_APR_MEASURE_READ=never")

  # Includes a shared fixture, which resolves from the `tests` root.
  prosper_add_gtest(test_ampr_measure
    SOURCES tests/hle/test_ampr_measure.cpp
    INCLUDES tests
    LIBRARIES prosper_core)

  # sceAmprCommandBufferGetSize must not answer for one command buffer with another's recorded
  # capacity: a too-small answer permanently hangs the guest's IoStore append loop, which polls
  # GetSize(cb) - GetCurrentOffset(cb) > 0xff without a bound (The Forgotten City, PPSA03026).
  prosper_add_gtest(test_ampr_getsize_fallback
    SOURCES tests/hle/test_ampr_getsize_fallback.cpp
    LIBRARIES prosper_core)

  # Direct-memory allocate/available (issue #99): AvailableDirectMemorySize reports the largest
  # free aligned block in the search window and writes both out-params (was a total-size stub).
  prosper_add_gtest(test_dmem
    SOURCES tests/host/test_dmem.cpp
    LIBRARIES prosper_core)

  # A refused fixed direct-memory map must preserve the allocation for a later retry.
  prosper_add_gtest(test_dmem_map_failure_rollback
    SOURCES tests/hle/memory/test_dmem_map_failure_rollback.cpp
    LIBRARIES prosper_core)

  # BatchMap is prefix-committing: a later refused mapping reports how many prior entries took
  # effect. Keep the failure count and retained direct mapping contract independently executable.
  prosper_add_gtest(test_batchmap_prefix_failure
    SOURCES tests/hle/memory/test_batchmap_prefix_failure.cpp
    LIBRARIES prosper_core)

  # MapDirectMemory2 changes physical-allocation type only when its mapping succeeds; exercise the
  # refused fixed-map path against a live host mapping and pair it with a successful type control.
  prosper_add_gtest(test_map_direct_memory2_failure
    SOURCES tests/hle/memory/test_map_direct_memory2_failure.cpp
    LIBRARIES prosper_core)

  # PROSPER_DMEM_CALLER stack walk (issue #1755): the diagnostic scans up to 160 slots above its
  # frame and used to read straight off the top of a thread stack, dying with a SIGSEGV that reads
  # as a guest fault.
  prosper_add_gtest(test_dmem_caller_walk
    SOURCES tests/test_dmem_caller_walk.cpp
    LIBRARIES prosper_core)

  # PROSPER_DMEM_CALLER correlation (issue #1859): the full stack remains one bounded line per
  # caller identity while every MEMLOG allocation can carry the same stable run-local ID. Pure,
  # including explicit unknown/overflow states and concurrent interning.
  # Does not link prosper_core, so it does not inherit its PUBLIC `src`.
  prosper_add_gtest(test_dmem_caller_chain
    SOURCES tests/hle/memory/test_dmem_caller_chain.cpp
    INCLUDES src)

  # #3057: test_dmem_caller_chain above covers format_dmem_caller_chain_definition (the formatter)
  # only -- it never calls k_alloc_dmem/k_alloc_main_dmem, so deleting the actual
  # attribute_dmem_allocation(...) call sites #3055 added left that suite green. These two bind to
  # the real call sites through the NID dispatch table, one process each (see the header comment in
  # the first file for why they cannot share a process).
  add_executable(test_dmem_caller_attribution_alloc_dmem
                 tests/hle/memory/test_dmem_caller_attribution_alloc_dmem.cpp)
  target_link_libraries(test_dmem_caller_attribution_alloc_dmem PRIVATE prosper_core)
  add_test(NAME dmem_caller_attribution_alloc_dmem
           COMMAND test_dmem_caller_attribution_alloc_dmem)

  add_executable(test_dmem_caller_attribution_alloc_main_dmem
                 tests/hle/memory/test_dmem_caller_attribution_alloc_main_dmem.cpp)
  target_link_libraries(test_dmem_caller_attribution_alloc_main_dmem PRIVATE prosper_core)
  add_test(NAME dmem_caller_attribution_alloc_main_dmem
           COMMAND test_dmem_caller_attribution_alloc_main_dmem)

  # Capped-diagnostic rate limit (issue #1761): a print budget must never be readable as a
  # population. Pins the sparse tail and the per-key budget; header-only, no GPU state.
  prosper_add_gtest(test_diag_ratelimit
    SOURCES tests/gpu/diagnostics/test_diag_ratelimit.cpp
    INCLUDES src)

  # The build journal must not crash on an unreadable label or present a partial/old
  # sample as an observed fence value. CPU-only: no command processor or Vulkan device.
  prosper_add_gtest(test_fence_build_journal
    SOURCES tests/gpu/diagnostics/test_fence_build_journal.cpp
            src/gpu/diagnostics/fence_build_journal.cpp src/host/memory/guest_memory_copy.cpp
    INCLUDES src)

  # Readback-reason policy: the verdict is expressed in terms of the reason, so a future edit
  # cannot change what is counted without changing what is decided.
  target_sources(test_cpu_rtt_snapshot_pool PRIVATE src/diagnostics/readback_reason_census.cpp)

  # Transfer pressure: host bytes/second, default-on. Pins that categories stay separable, that a
  # zero-byte call is not an event, and that an out-of-range category cannot corrupt a neighbour.
  prosper_add_gtest(test_transfer_pressure
    SOURCES tests/gpu/diagnostics/test_transfer_pressure.cpp src/diagnostics/transfer_pressure.cpp
    src/diagnostics/exit_census.cpp src/diagnostics/exit_reports.cpp
    INCLUDES src)

  # Classified draw disposition: six involuntary drop paths folded into one counter, two of them
  # silent. Pins that a healthy pass stays quiet, that a drop names its mechanism, and that the
  # census reports an unnamed drop path as UNACCOUNTED rather than absorbing it.
  prosper_add_gtest(test_draw_disposition
    SOURCES tests/gpu/diagnostics/test_draw_disposition.cpp src/gpu/diagnostics/draw_disposition.cpp src/diagnostics/exit_census.cpp src/diagnostics/exit_reports.cpp
    INCLUDES src)

  # Pass-break classification: keeps the per-reason distribution separable, because the reason
  # that ends the most passes is not the reason holding the mean down.
  prosper_add_gtest(test_pass_break_census
    SOURCES tests/gpu/diagnostics/test_pass_break_census.cpp src/gpu/diagnostics/pass_break_census.cpp
    src/diagnostics/exit_census.cpp src/diagnostics/exit_reports.cpp
    INCLUDES src)

  # IME keyboard event delivery (issue #1093): sceImeUpdate must invoke the guest handler
  # per queued key event; the old stub delivered none.
  # Includes a shared fixture, which resolves from the `tests` root.
  prosper_add_gtest(test_ime_input
    SOURCES tests/hle/input/test_ime_input.cpp
    INCLUDES tests
    LIBRARIES prosper_core)

  # Huge non-fixed sceKernelReserveVirtualRange placement (#312/#946): the 512 GiB UE arena
  # must be steered off its low hint into the guest auto window (see test for the contract).
  prosper_add_gtest(test_reserve_placement
    SOURCES tests/hle/test_reserve_placement.cpp
    LIBRARIES prosper_core)

  # A protection-only change must retain an uncommitted reservation's state (#343), or guest
  # VirtualQuery skips the later commit and the first access faults. Cross-platform HLE contract.
  prosper_add_gtest(test_retrack_reservation
    SOURCES tests/hle/test_retrack_reservation.cpp
    LIBRARIES prosper_core)

  prosper_add_gtest(test_guest_memory_topology
    SOURCES tests/hle/memory/test_guest_memory_topology.cpp
    LIBRARIES prosper_core)
  add_executable(test_guest_mapping_lease tests/hle/memory/test_guest_mapping_lease.cpp)
  target_link_libraries(test_guest_mapping_lease PRIVATE prosper_core)
  add_test(NAME guest_mapping_lease COMMAND test_guest_mapping_lease)
  set_tests_properties(guest_mapping_lease PROPERTIES TIMEOUT 20)
  if(WIN32)
    prosper_add_gtest(test_windows_direct_fault_safety
      SOURCES tests/hle/memory/test_windows_direct_fault_safety.cpp
      LIBRARIES prosper_core)
  endif()
  prosper_add_gtest(test_nested_wide_data_admission
    SOURCES tests/gpu/execute/test_nested_wide_data_admission.cpp
    LIBRARIES prosper_core)
  # Registered nested snapshot admission uses the Linux direct-memory lazy-fault proof.
  # Callback/compute stand-ins exercise ordered execution without creating a Vulkan device.
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    prosper_add_gtest(test_ordered_render_finality
      SOURCES tests/gpu/execute/test_ordered_render_finality.cpp
      INCLUDES frontends
      LIBRARIES prosper_core)
  endif()

  # Hinted map must not clobber a live mapping (#137): MAP_FIXED_NOREPLACE unless the hint is our
  # own uncommitted reservation. Drives the real map HLE handlers. Linux-only (mmap semantics).
  if(UNIX)
    prosper_add_gtest(test_map_noreplace
      SOURCES tests/hle/test_map_noreplace.cpp
      LIBRARIES prosper_core)

    # A guest mprotect(prot=0) makes the page PROT_NONE, not a default RW (#342): guard pages must
    # actually guard. Drives the real sceKernelMprotect + a SIGSEGV probe. Linux-only.
    prosper_add_gtest(test_prot_none
      SOURCES tests/hle/test_prot_none.cpp
      LIBRARIES prosper_core)
  endif()

  # PM4 command-stream decoder (front of the CommandProcessor): decode a Dcb built by the AGC
  # Dcb functions and assert the ops. Pure/cross-platform, no game dump needed.
  prosper_add_gtest(test_pm4_decode
    SOURCES tests/gpu/pm4/test_pm4_decode.cpp
    LIBRARIES prosper_core)

  # CommandProcessor apply stage: replay a decoded stream into a GpuState (register files + draws).
  add_executable(test_command_processor tests/gpu/pm4/test_command_processor.cpp)
  target_link_libraries(test_command_processor PRIVATE prosper_core)
  add_test(NAME command_processor_apply COMMAND test_command_processor)

  add_executable(test_command_provenance tests/gpu/pm4/test_command_provenance.cpp)
  target_link_libraries(test_command_provenance PRIVATE prosper_core)
  add_test(NAME command_provenance COMMAND test_command_provenance)

  # Stage B: CommandProcessor honors RELEASE_MEM/WRITE_DATA memory writes (synchronous end-of-pipe).
  prosper_add_gtest(test_eop_write
    SOURCES tests/gpu/test_eop_write.cpp
    LIBRARIES prosper_core)

  # #4003: force the stale-drainer handoff in the actual completion-queue implementation.
  # This executable supplies its own CP object; the ordinary archive CP object stays unselected.
  # Rendezvous calls are compiled out of prosper_core and every shipping frontend.
  add_executable(test_pending_drain_gate tests/gpu/pm4/test_pending_drain_gate.cpp
    src/gpu/pm4/command_processor.cpp)
  target_link_libraries(test_pending_drain_gate PRIVATE prosper_core)
  target_compile_definitions(test_pending_drain_gate PRIVATE PROSPER_PENDING_DRAIN_TEST_CHECKPOINTS)
  if(PROSPER_DIAGNOSTICS)
    target_compile_definitions(test_pending_drain_gate PRIVATE PROSPER_DIAGNOSTICS)
  endif()
  if(UNIX AND NOT APPLE)
    target_link_options(test_pending_drain_gate PRIVATE
      "-Wl,-Map,${CMAKE_CURRENT_BINARY_DIR}/test_pending_drain_gate.map")
  endif()
  add_test(NAME pending_drain_gate COMMAND test_pending_drain_gate)
  add_test(NAME pending_drain_legacy COMMAND test_pending_drain_gate --legacy)
  set_tests_properties(pending_drain_gate pending_drain_legacy PROPERTIES TIMEOUT 15)

  # #1945: the interior free-chain poison scan — its structural predicate, and the census that keeps
  # a zero from reading as "the chains are clean" when the walk reached nothing. POSIX only: the test
  # needs a MAP_FIXED anonymous mapping inside the guest VA window to build a pool array with real
  # addresses, and mingw-w64 has no <sys/mman.h>. The code under test is built on every platform.
  if(NOT WIN32)
    add_executable(test_mb3_poison_scan tests/gpu/execute/test_mb3_poison_scan.cpp)
    target_link_libraries(test_mb3_poison_scan PRIVATE prosper_core)
    add_test(NAME mb3_poison_scan COMMAND test_mb3_poison_scan)
    # A host that cannot place the mapping must SKIP, not pass: the structural half of the test
    # would otherwise be silently absent from a green run.
    set_tests_properties(mb3_poison_scan PROPERTIES SKIP_RETURN_CODE 77)
  endif()

  # #729: the synchronous (PROSPER_EOP_WRITE_SYNC=1) honor_* label writes must probe
  # guest_readable before dereferencing a guest-PM4-supplied address (own executable: the
  # env var latches process-wide on first use).
  prosper_add_gtest(test_eop_write_sync_guard
    SOURCES tests/gpu/pm4/test_eop_write_sync_guard.cpp
    LIBRARIES prosper_core)

  # #312 per-queue WAIT_REG_MEM barrier model: unsatisfied waits gate downstream memory effects,
  # streams release independently, jumps don't un-gate, timeout backstop preserves liveness.
  prosper_add_gtest(test_wait_barrier
    SOURCES tests/gpu/pm4/test_wait_barrier.cpp
    LIBRARIES prosper_core)
  add_executable(test_wait_regmem_observation tests/diagnostics/test_wait_regmem_observation.cpp)
  target_link_libraries(test_wait_regmem_observation PRIVATE prosper_core)
  add_test(NAME wait_regmem_observation COMMAND test_wait_regmem_observation
    "${CMAKE_CURRENT_BINARY_DIR}/wait-regmem-observation.jsonl")
  add_test(NAME wait_regmem_observation_disabled COMMAND test_wait_regmem_observation
    "${CMAKE_CURRENT_BINARY_DIR}/wait-regmem-observation-disabled.jsonl")
  set_tests_properties(wait_regmem_observation_disabled PROPERTIES
    ENVIRONMENT "PROSPER_NO_PERF_ALARMS=1")
  add_executable(test_wait_regmem_packet_route tests/diagnostics/test_wait_regmem_packet_route.cpp)
  target_link_libraries(test_wait_regmem_packet_route PRIVATE prosper_core)
  add_test(NAME wait_regmem_packet_route COMMAND test_wait_regmem_packet_route
    "${CMAKE_CURRENT_BINARY_DIR}/wait-regmem-packet-route.jsonl" proceed)
  add_test(NAME wait_regmem_packet_route_defer COMMAND test_wait_regmem_packet_route
    "${CMAKE_CURRENT_BINARY_DIR}/wait-regmem-packet-route-defer.jsonl" defer)
  add_test(NAME wait_regmem_packet_route_disabled COMMAND test_wait_regmem_packet_route
    "${CMAKE_CURRENT_BINARY_DIR}/wait-regmem-packet-route-disabled.jsonl" proceed)
  set_tests_properties(wait_regmem_packet_route_disabled PROPERTIES
    ENVIRONMENT "PROSPER_NO_PERF_ALARMS=1")
  set_tests_properties(wait_regmem_packet_route wait_regmem_packet_route_defer
    wait_regmem_packet_route_disabled PROPERTIES TIMEOUT 30)

  # Render-state extraction: fold a stream into a GpuState, then interpret RDNA2 registers.
  prosper_add_gtest(test_render_state
    SOURCES tests/gpu/state/test_render_state.cpp
    LIBRARIES prosper_core)

  # Messenger recompile guard (#228): committed real Messenger VS/PS RDNA2 binaries must keep
  # recompiling — a recompiler change that re-rejects one of their instructions (the #199/#201
  # blue-screen class) fails here instead of shipping. Pure (no Vulkan); needs no game dump.
  prosper_add_gtest(test_messenger_recompile_guard
    SOURCES tests/gpu/recompiler/test_messenger_recompile_guard.cpp
    LIBRARIES prosper_core)
  target_compile_definitions(test_messenger_recompile_guard PRIVATE
      PROSPER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/data")

  # libkernel/libScePosix signal + time-conversion surface (#190): NID match, registration, and real
  # behavior (UTC<->localtime round-trip fills its out-param; pthread_setcancelstate old-state + EINVAL).
  prosper_add_gtest(test_kernel_signal_time
    SOURCES tests/hle/test_kernel_signal_time.cpp
    LIBRARIES prosper_core)

  # Pipeline-state resolution: RenderState (RDNA2 enums) -> Vulkan-ready pipeline state (pure, no Vulkan).
  prosper_add_gtest(test_pipeline_state
    SOURCES tests/gpu/state/test_pipeline_state.cpp
    LIBRARIES prosper_core)

  # Recompiler coverage metric (pure): per-instruction support report used by shader_histo.
  add_executable(test_recompile_coverage tests/gpu/test_recompile_coverage.cpp)
  target_link_libraries(test_recompile_coverage PRIVATE prosper_core)
  add_test(NAME recompile_coverage COMMAND test_recompile_coverage)

  # Pure CFG proof regression: compile only the decoder, so adversarial barrier-uniformity
  # cases can be checked without a renderer/core build or a potentially unsafe GPU dispatch.
  prosper_add_gtest(test_workgroup_uniformity
    SOURCES tests/gpu/test_workgroup_uniformity.cpp src/gpu/recompiler/rdna2_decode.cpp
    INCLUDES src)

  prosper_add_gtest(test_direct_descriptor_copy
    SOURCES tests/gpu/recompiler/test_direct_descriptor_copy.cpp
    LIBRARIES prosper_core)

  prosper_add_gtest(test_smem_pointer_provenance
    SOURCES tests/gpu/recompiler/test_smem_pointer_provenance.cpp
    LIBRARIES prosper_core)

  prosper_add_gtest(test_image_descriptor_class_lookup
    SOURCES tests/gpu/recompiler/test_image_descriptor_class_lookup.cpp
    LIBRARIES prosper_core)

  # #657: query-only descriptors. The negative control (one added sample instruction) is what makes
  # the positive arm mean anything, so both live in one binary.
  prosper_add_gtest(test_query_only_binding
    SOURCES tests/gpu/test_query_only_binding.cpp
    LIBRARIES prosper_core)

  # Bounded graphics-shader cache: cached modules must match the direct recompiler byte-for-byte.
  # The trip-bound OPERATION scope (#3714): one sampled selector value shared by the shader-cache
  # key and the compilation it names. Needs no GPU and no fixture -- it asserts agreement between
  # two reads, plus a work counter, because correct settings cannot show that the sampling happened
  # once rather than four times.
  prosper_add_gtest(test_trip_bound_operation
    SOURCES tests/gpu/execute/test_trip_bound_operation.cpp
    LIBRARIES prosper_core)

  # #3745: CachedShader's hand-written copy pair suppressed its implicit moves, so every cache
  # insertion copied. Asserts the move pair exists and transfers the SPIR-V shared_ptr.
  prosper_add_gtest(test_cached_shader_moves
    SOURCES tests/gpu/execute/test_cached_shader_moves.cpp
    LIBRARIES prosper_core)

  add_executable(test_shader_recompile_cache tests/gpu/execute/test_shader_recompile_cache.cpp)
  add_executable(fragment_compile_replay tools/fragment_compile_replay/main.cpp)
  target_link_libraries(fragment_compile_replay PRIVATE prosper_core)
  add_executable(test_fragment_compile_case tests/gpu/execute/test_fragment_compile_case.cpp)
  target_include_directories(test_fragment_compile_case PRIVATE tests)
  target_link_libraries(test_fragment_compile_case PRIVATE prosper_core)
  add_test(NAME fragment_compile_case COMMAND test_fragment_compile_case)
  if(Python3_Interpreter_FOUND)
    add_test(NAME fragment_compile_case_cli COMMAND "${Python3_EXECUTABLE}"
      "${CMAKE_SOURCE_DIR}/tools/fragment_compile_replay/test_cli.py"
      "$<TARGET_FILE:test_fragment_compile_case>" "$<TARGET_FILE:fragment_compile_replay>")
  endif()
  # Includes a shared fixture, which resolves from the `tests` root.
  target_include_directories(test_shader_recompile_cache PRIVATE tests)
  target_link_libraries(test_shader_recompile_cache PRIVATE prosper_core)
  add_test(NAME shader_recompile_cache COMMAND test_shader_recompile_cache)
  # Each policy is frozen for the process lifetime; exercise all combinations in separate tests.
  foreach(key_policy IN ITEMS legacy_hash legacy_scratch legacy_both)
    add_test(NAME shader_recompile_cache_${key_policy} COMMAND test_shader_recompile_cache)
  endforeach()
  set_tests_properties(shader_recompile_cache_legacy_hash PROPERTIES
    ENVIRONMENT "PROSPER_NO_SHADER_KEY_WORD_HASH=1")
  set_tests_properties(shader_recompile_cache_legacy_scratch PROPERTIES
    ENVIRONMENT "PROSPER_NO_SHADER_KEY_SCRATCH=1")
  set_tests_properties(shader_recompile_cache_legacy_both PROPERTIES
    ENVIRONMENT "PROSPER_NO_SHADER_KEY_WORD_HASH=1;PROSPER_NO_SHADER_KEY_SCRATCH=1")

  # RDNA2 shader instruction-stream walker (first stage of the RDNA2->SPIR-V recompiler).
  prosper_add_gtest(test_compressed_source_authority
    SOURCES tests/gpu/resources/test_compressed_source_authority.cpp
    LIBRARIES prosper_core)

  prosper_add_gtest(test_metadata_kind_correlation
    SOURCES tests/gpu/resources/test_metadata_kind_correlation.cpp
    LIBRARIES prosper_core)

  prosper_add_gtest(test_watch_list
    SOURCES tests/gpu/diagnostics/test_watch_list.cpp
    LIBRARIES prosper_core)

  # PROSPER_SKIP_DRAW_PROGRAM: decline graphics draws by shader PROGRAM identity (the graphics
  # counterpart of PROSPER_COMPUTE_SKIP_PROGRAM). Pure selector logic -- no Vulkan; the wiring into
  # the live renderer is pinned separately by draw_program_skip_render below.
  prosper_add_gtest(test_draw_program_skip
    SOURCES tests/gpu/diagnostics/test_draw_program_skip.cpp
    LIBRARIES prosper_core)

  # PROSPER_DRAW_LINKSCAN: the CPU-side census of the linked lists a draw's scalar buffers contain
  # (the graphics counterpart of PROSPER_COMPUTE_PARENTSCAN). Pure analysis + selector -- no Vulkan.
  prosper_add_gtest(test_link_list_census
    SOURCES tests/gpu/diagnostics/test_link_list_census.cpp
    LIBRARIES prosper_core)

  # #4273: each distinct refused shader is kept by default, bounded and deduplicated.
  prosper_add_gtest(test_refused_shader_dump
    SOURCES tests/gpu/diagnostics/test_refused_shader_dump.cpp
    INCLUDES tests
    LIBRARIES prosper_core)

  prosper_add_gtest(test_shader_dump_filter
    SOURCES tests/gpu/diagnostics/test_shader_dump_filter.cpp
    LIBRARIES prosper_core)

  # PROSPER_GPU_BREADCRUMBS core + emitter against recording Vulkan entry points. No device: stock
  # lavapipe exposes neither marker extension, so what this proves is the emitter/resolver contract,
  # not that a real driver executes the writes before a hang (a local run, recorded in the PR).
  # Both tests include <vulkan/vulkan.h> through their headers, so they need Vulkan: MinGW and macOS
  # runners have no Vulkan headers and failed to compile them when this was unconditional.
  if(NOT TARGET Vulkan::Vulkan)
    find_package(Vulkan QUIET)
  endif()
  if(TARGET Vulkan::Vulkan)
  prosper_add_gtest(test_gpu_breadcrumbs
    SOURCES tests/gpu/diagnostics/test_gpu_breadcrumbs.cpp
    LIBRARIES prosper_core Vulkan::Vulkan)

  # PROSPER_GPU_LABELS: the label text, colour class and begin/end pairing, against recording
  # debug-utils entry points. What a capture tool DISPLAYS of them is a local look, not a ctest.
  prosper_add_gtest(test_gpu_labels
    SOURCES tests/gpu/diagnostics/test_gpu_labels.cpp
    LIBRARIES prosper_core Vulkan::Vulkan)
  endif()

  prosper_add_gtest(test_gpu_memory_budget
    SOURCES tests/gpu/diagnostics/test_gpu_memory_budget.cpp
    INCLUDES tests
    LIBRARIES prosper_core)

  # #3873: the persistent texture cache's budget from VK_EXT_memory_budget headroom (pure policy).
  prosper_add_gtest(test_texture_cache_budget
    SOURCES tests/gpu/memory/test_texture_cache_budget.cpp
    LIBRARIES prosper_core)

  # #3905: when spilled retained resources may be evicted to be re-created device-local (pure).
  prosper_add_gtest(test_spill_recovery
    SOURCES tests/gpu/memory/test_spill_recovery.cpp
    LIBRARIES prosper_core)

  # #3888: GPU-only images/buffers prefer a DEVICE_LOCAL memory type (pure; Vulkan headers only).
  if(TARGET Vulkan::Vulkan)
    prosper_add_gtest(test_memory_type_select
      SOURCES tests/gpu/memory/test_memory_type_select.cpp
      LIBRARIES prosper_core Vulkan::Vulkan)
  endif()

  prosper_add_gtest(test_rdna2_decode
    SOURCES tests/gpu/recompiler/test_rdna2_decode.cpp
    LIBRARIES prosper_core)

  prosper_add_gtest(test_rdna2_predicate_tables
    SOURCES tests/gpu/recompiler/test_rdna2_predicate_tables.cpp
    LIBRARIES prosper_core)

  # Exhaustive per-family opcode/length sweep of the walker (words built from the encoding layout).
  prosper_add_gtest(test_rdna2_decode_sweep
    SOURCES tests/gpu/recompiler/test_rdna2_decode_sweep.cpp
    LIBRARIES prosper_core)

  # Structural RDNA2->SPIR-V validation that does not require Vulkan runtime support.
  add_executable(test_rdna2_spirv_struct tests/gpu/test_rdna2_spirv_struct.cpp)
  target_link_libraries(test_rdna2_spirv_struct PRIVATE prosper_core)
  add_test(NAME rdna2_spirv_struct COMMAND test_rdna2_spirv_struct)

  prosper_add_gtest(test_compute_dpp_row_max
    SOURCES tests/gpu/recompiler/test_compute_dpp_row_max.cpp
    INCLUDES tests
    LIBRARIES prosper_core)
  prosper_add_gtest(test_compute_dpp_row_fadd
    SOURCES tests/gpu/recompiler/test_compute_dpp_row_fadd.cpp
    INCLUDES tests
    LIBRARIES prosper_core)
  if(Vulkan_FOUND)
    prosper_add_gtest(test_compute_dpp_row_fadd_execution
      SOURCES tests/gpu/execute/test_compute_dpp_row_fadd.cpp
      INCLUDES tests
      LIBRARIES prosper_core Vulkan::Vulkan
      PROPERTIES TIMEOUT 60)
    prosper_add_gtest(test_compute_dpp_row_max_execution
      SOURCES tests/gpu/execute/test_compute_dpp_row_max.cpp
      INCLUDES tests
      LIBRARIES prosper_core Vulkan::Vulkan
      PROPERTIES TIMEOUT 60)
  endif()

  # #4007: strict-valid, paired address/control counterexamples for the load-bearing width proof.
  add_executable(test_wave_width_independent tests/gpu/recompiler/test_wave_width_independent.cpp)
  target_link_libraries(test_wave_width_independent PRIVATE prosper_core)
  add_test(NAME wave_width_independent COMMAND test_wave_width_independent)

  prosper_add_gtest(test_fragment_vote_lowering
    SOURCES tests/gpu/recompiler/test_fragment_vote_lowering.cpp
    LIBRARIES prosper_core)
  add_executable(test_fragment_vote_refusal_details tests/gpu/recompiler/test_fragment_vote_refusal_details.cpp)
  target_link_libraries(test_fragment_vote_refusal_details PRIVATE prosper_core)
  add_test(NAME fragment_vote_refusal_details COMMAND test_fragment_vote_refusal_details)
  add_test(NAME fragment_vote_refusal_details_disabled COMMAND test_fragment_vote_refusal_details --disabled)
  set_tests_properties(fragment_vote_refusal_details_disabled PROPERTIES ENVIRONMENT "PROSPER_NO_PERF_ALARMS=1")
  add_executable(test_fragment_loop_vote_lowering tests/gpu/recompiler/test_fragment_loop_vote_lowering.cpp)
  target_link_libraries(test_fragment_loop_vote_lowering PRIVATE prosper_core)
  add_test(NAME fragment_loop_vote_lowering COMMAND test_fragment_loop_vote_lowering)
  add_executable(test_fragment_neutral_selection tests/gpu/recompiler/test_fragment_neutral_selection.cpp)
  target_link_libraries(test_fragment_neutral_selection PRIVATE prosper_core)
  add_test(NAME fragment_neutral_selection COMMAND test_fragment_neutral_selection)

  # #4016: exact compare-mask SSA, CMPX VCC preservation, and strict scalar-AND controls.
  prosper_add_gtest(test_compare_exec_masking
    SOURCES tests/gpu/recompiler/test_compare_exec_masking.cpp
    LIBRARIES prosper_core)

  add_executable(test_normal_magnitude_predicate tests/gpu/recompiler/test_normal_magnitude_predicate.cpp)
  target_link_libraries(test_normal_magnitude_predicate PRIVATE prosper_core)
  add_test(NAME normal_magnitude_predicate COMMAND test_normal_magnitude_predicate)
  add_executable(test_raster_quad_collector tests/gpu/recompiler/test_raster_quad_collector.cpp)
  target_link_libraries(test_raster_quad_collector PRIVATE prosper_core)
  add_test(NAME raster_quad_collector_contract COMMAND test_raster_quad_collector)
  add_executable(test_fragment_float_mode tests/gpu/recompiler/test_fragment_float_mode.cpp)
  target_link_libraries(test_fragment_float_mode PRIVATE prosper_core)
  target_include_directories(test_fragment_float_mode PRIVATE tests)
  add_test(NAME fragment_float_mode COMMAND test_fragment_float_mode)

  add_executable(test_fragment_float_transport tests/gpu/recompiler/test_fragment_float_transport.cpp)
  target_link_libraries(test_fragment_float_transport PRIVATE prosper_core)
  add_test(NAME fragment_float_transport COMMAND test_fragment_float_transport)
  add_executable(test_fragment_arithmetic_diagnostics tests/gpu/recompiler/test_fragment_arithmetic_diagnostics.cpp)
  target_link_libraries(test_fragment_arithmetic_diagnostics PRIVATE prosper_core)
  target_include_directories(test_fragment_arithmetic_diagnostics PRIVATE tests)
  add_test(NAME fragment_arithmetic_diagnostics COMMAND test_fragment_arithmetic_diagnostics)

  # #3203: the entry-M0 token containment's CFG-DISPATCHER re-arm. Its own binary rather than another
  # arm in test_rdna2_spirv_struct, because that one returns at its first failure and a mutation run
  # then cannot tell which arms moved. This reports every arm independently.
  prosper_add_gtest(test_entry_m0_dispatcher
    SOURCES tests/gpu/recompiler/test_entry_m0_dispatcher.cpp
    LIBRARIES prosper_core)

  prosper_add_gtest(test_scalar_ff1_b32
    SOURCES tests/gpu/recompiler/test_scalar_ff1_b32.cpp
    LIBRARIES prosper_core)

  # #3231: the CFG dispatcher's `missing-entry-vcc` gate and the entry-block-dominance case it may
  # now admit. Both directions, in its own binary for the same reason as the file above.
  prosper_add_gtest(test_entry_vcc_dead
    SOURCES tests/gpu/recompiler/test_entry_vcc_dead.cpp
    LIBRARIES prosper_core)

  # Scalar const-fold interpreter behind bindless-dynamic vertex-fetch resolution (no Vulkan needed).
  add_executable(test_dynfetch_fold tests/gpu/recompiler/test_dynfetch_fold.cpp)
  # Includes a shared fixture, which resolves from the `tests` root.
  target_include_directories(test_dynfetch_fold PRIVATE tests)
  target_link_libraries(test_dynfetch_fold PRIVATE prosper_core)
  add_test(NAME dynfetch_fold COMMAND test_dynfetch_fold)
  add_executable(test_raw_register_wide_data tests/gpu/recompiler/test_raw_register_wide_data.cpp)
  target_link_libraries(test_raw_register_wide_data PRIVATE prosper_core)
  add_test(NAME raw_register_wide_data COMMAND test_raw_register_wide_data)
  prosper_add_gtest(test_raw_wide_flat_read_effects
    SOURCES tests/gpu/recompiler/test_raw_wide_flat_read_effects.cpp
            src/gpu/recompiler/rdna2_decode.cpp
            src/gpu/recompiler/rdna2_raw_wide_data.cpp
    INCLUDES src)
  prosper_add_gtest(test_graphics_raw_register_wide_data
    SOURCES tests/gpu/recompiler/test_graphics_raw_register_wide_data.cpp
    LIBRARIES prosper_core)
  add_executable(test_raw_wide_mask_lifetime
    tests/gpu/recompiler/test_raw_wide_mask_lifetime.cpp)
  target_link_libraries(test_raw_wide_mask_lifetime PRIVATE prosper_core)
  add_test(NAME raw_wide_mask_lifetime COMMAND test_raw_wide_mask_lifetime)
  prosper_add_gtest(test_owned_raw_x2
    SOURCES tests/gpu/execute/test_owned_raw_x2.cpp
    LIBRARIES prosper_core)
  prosper_add_gtest(test_raw_wide_read_point
    SOURCES tests/gpu/recompiler/test_raw_wide_read_point.cpp
    LIBRARIES prosper_core)
  add_executable(test_owned_raw_wide_backing tests/gpu/recompiler/test_owned_raw_wide_backing.cpp)
  target_link_libraries(test_owned_raw_wide_backing PRIVATE prosper_core)
  add_test(NAME owned_raw_wide_backing COMMAND test_owned_raw_wide_backing)
  prosper_add_gtest(test_memory_fed_raw_wide
    SOURCES tests/gpu/recompiler/test_memory_fed_raw_wide.cpp
    LIBRARIES prosper_core)
  add_executable(test_folded_stride_zero tests/gpu/recompiler/test_folded_stride_zero.cpp)
  target_link_libraries(test_folded_stride_zero PRIVATE prosper_core)
  add_test(NAME folded_stride_zero COMMAND test_folded_stride_zero)
  # The ordinary test checks selected-dispatch results. This second arm proves the opt-in
  # stage-fold profiler actually counts that selected path, so a zero game census is meaningful.
  if(Python3_Interpreter_FOUND)
    add_test(NAME dynfetch_fold_profile COMMAND "${Python3_EXECUTABLE}"
      "${CMAKE_SOURCE_DIR}/tests/gpu/recompiler/test_dynfetch_fold_profile.py"
      "$<TARGET_FILE:test_dynfetch_fold>")
  endif()
  add_executable(fold_replay tools/fold_replay/main.cpp)
target_link_libraries(fold_replay PRIVATE prosper_core)

add_executable(test_fold_capture tests/gpu/execute/test_fold_capture.cpp)
  target_link_libraries(test_fold_capture PRIVATE prosper_core)
  add_test(NAME fold_capture COMMAND test_fold_capture)
if(Python3_Interpreter_FOUND)
  add_test(NAME fold_capture_cli COMMAND "${Python3_EXECUTABLE}"
    "${CMAKE_SOURCE_DIR}/tools/fold_replay/test_cli.py"
    "$<TARGET_FILE:test_fold_capture>" "$<TARGET_FILE:fold_replay>")
endif()

  add_executable(test_fold_control_plan tests/gpu/execute/test_fold_control_plan.cpp)
  target_link_libraries(test_fold_control_plan PRIVATE prosper_core)
  add_test(NAME fold_control_plan COMMAND test_fold_control_plan)
  add_test(NAME fold_control_plan_uncached COMMAND test_fold_control_plan)
  set_tests_properties(fold_control_plan_uncached PROPERTIES
    ENVIRONMENT "PROSPER_NO_FOLD_CONTROL_CACHE=1")

  # #3137: the FORMAT-path MUBUF reject diagnostic must name which provenance step failed.
  prosper_add_gtest(test_vertex_fetch_reject_diagnostic
    SOURCES tests/gpu/recompiler/test_vertex_fetch_reject_diagnostic.cpp
    LIBRARIES prosper_core)

  # #3596: a lane-local scalar must not satisfy the wave-uniformity proof.
  prosper_add_gtest(test_readfirstlane_uniformity
    SOURCES tests/gpu/recompiler/test_readfirstlane_uniformity.cpp
    LIBRARIES prosper_core)

  add_executable(test_portable_bpermute tests/gpu/recompiler/test_portable_bpermute.cpp)
  target_include_directories(test_portable_bpermute PRIVATE tests)
  target_link_libraries(test_portable_bpermute PRIVATE prosper_core)
  add_test(NAME portable_bpermute_contract COMMAND test_portable_bpermute)

  add_executable(test_fragment_packet tests/gpu/recompiler/test_fragment_packet.cpp)
  target_include_directories(test_fragment_packet PRIVATE tests)
  target_link_libraries(test_fragment_packet PRIVATE prosper_core)
  add_test(NAME fragment_packet_contract COMMAND test_fragment_packet)
  prosper_add_gtest(test_fragment_resource_packet
    SOURCES tests/gpu/recompiler/test_fragment_resource_packet.cpp
    INCLUDES tests LIBRARIES prosper_core PROPERTIES TIMEOUT 120)
  prosper_add_gtest(test_fragment_packet_definedness
    SOURCES tests/gpu/recompiler/test_fragment_packet_definedness.cpp
    INCLUDES tests LIBRARIES prosper_core PROPERTIES TIMEOUT 120)
  prosper_add_gtest(test_fragment_packet_raw_masks
    SOURCES tests/gpu/recompiler/test_fragment_packet_raw_masks.cpp
    INCLUDES tests LIBRARIES prosper_core PROPERTIES TIMEOUT 120)
  prosper_add_gtest(test_fragment_packet_mask_entry
    SOURCES tests/gpu/recompiler/test_fragment_packet_mask_entry.cpp
    INCLUDES tests LIBRARIES prosper_core PROPERTIES TIMEOUT 120)
  prosper_add_gtest(test_fragment_special_f32
    SOURCES tests/gpu/recompiler/test_fragment_special_f32.cpp
    INCLUDES tests LIBRARIES prosper_core PROPERTIES TIMEOUT 120)
  prosper_add_gtest(test_fragment_packet_wave_data
    SOURCES tests/gpu/recompiler/test_fragment_packet_wave_data.cpp
    INCLUDES tests LIBRARIES prosper_core PROPERTIES TIMEOUT 120)
  prosper_add_gtest(test_fragment_packet_exports
    SOURCES tests/gpu/recompiler/test_fragment_packet_exports.cpp
    INCLUDES tests tests/gpu/recompiler LIBRARIES prosper_core PROPERTIES TIMEOUT 120)
  prosper_add_gtest(test_fragment_packet_waitcnt
    SOURCES tests/gpu/recompiler/test_fragment_packet_waitcnt.cpp
    INCLUDES tests tests/gpu/recompiler LIBRARIES prosper_core PROPERTIES TIMEOUT 120)
  add_executable(test_fragment_packet_wqm tests/gpu/recompiler/test_fragment_packet_wqm.cpp)
  target_include_directories(test_fragment_packet_wqm PRIVATE tests)
  target_link_libraries(test_fragment_packet_wqm PRIVATE prosper_core)
  add_test(NAME fragment_packet_wqm_contract COMMAND test_fragment_packet_wqm)
  # TIMEOUT 300, not 90: measured 62-74 s on Linux, ~70 s on Windows MinGW, and >90 s on the
  # macOS x86_64/Rosetta 2 job, where it has timed out on every run (#4176). Same for mbcnt
  # below (89 s on Linux under -j4). A stopgap for the test's cost, not a fix -- #4176.
  set_tests_properties(fragment_packet_wqm_contract PROPERTIES TIMEOUT 300)
  add_executable(test_fragment_packet_mbcnt tests/gpu/recompiler/test_fragment_packet_mbcnt.cpp)
  target_include_directories(test_fragment_packet_mbcnt PRIVATE tests)
  target_link_libraries(test_fragment_packet_mbcnt PRIVATE prosper_core)
  add_test(NAME fragment_packet_mbcnt_contract COMMAND test_fragment_packet_mbcnt)
  set_tests_properties(fragment_packet_mbcnt_contract PROPERTIES TIMEOUT 300)
  set_tests_properties(portable_bpermute_contract PROPERTIES TIMEOUT 90)

  # #3713: a scalar branch may only be called workgroup-uniform when the value it compares is the
  # same for every wave along every path. Built as positive/negative pairs -- a proof that never
  # fires would otherwise pass the suite.
  prosper_add_gtest(test_scc_branch_uniformity
    SOURCES tests/gpu/recompiler/test_scc_branch_uniformity.cpp
    LIBRARIES prosper_core)

  prosper_add_gtest(test_fragment_draw_capacity
    SOURCES tests/gpu/recompiler/test_fragment_draw_capacity.cpp
    INCLUDES src tests
    LIBRARIES prosper_core)

  # #3573: the CFG dispatcher must lower a fragment exec/vcc branch as a wave vote.
  prosper_add_gtest(test_dispatcher_wave_vote
    SOURCES tests/gpu/recompiler/test_dispatcher_wave_vote.cpp
    LIBRARIES prosper_core)

  # #3577: a null image descriptor must carry the shape the shader declares.
  prosper_add_gtest(test_null_image_shape
    SOURCES tests/gpu/execute/test_null_image_shape.cpp
    LIBRARIES prosper_core)

  # #3579: a refused buffer instruction must not print the same census word as an emitted one.
  prosper_add_gtest(test_buf_op_census_disposition
    SOURCES tests/gpu/recompiler/test_buf_op_census_disposition.cpp
    LIBRARIES prosper_core)

  # #3479: a recompiled guest module must declare SignedZeroInfNanPreserve, or a driver is free to
  # compile it under no-Inf/no-NaN assumptions and silently answer a guest sign() idiom with 0 --
  # and must NOT declare it on a device that cannot execute it, which is invalid rather than merely
  # unoptimised (#3561). Both directions are asserted. Structural on purpose -- an execution arm
  # cannot go red where CI runs. See the test.
  prosper_add_gtest(test_float_controls
    SOURCES tests/gpu/recompiler/test_float_controls.cpp
    LIBRARIES prosper_core)

  # CPU-only streaming shader properties: exact legacy semantics without per-query vectors.
  add_executable(test_shader_property_scans tests/gpu/recompiler/test_shader_property_scans.cpp)
  target_link_libraries(test_shader_property_scans PRIVATE prosper_core)
  add_test(NAME shader_property_scans COMMAND ${CMAKE_COMMAND} -E env
           --unset=PROSPER_PROLOGLOG $<TARGET_FILE:test_shader_property_scans>)

  prosper_add_gtest(test_gta5_selected_sbuffer
    SOURCES tests/gpu/recompiler/test_gta5_selected_sbuffer.cpp
    LIBRARIES prosper_core)

  # Includes a shared fixture, which resolves from the `tests` root.
  prosper_add_gtest(test_gta5_packed_pointer
    SOURCES tests/gpu/test_gta5_packed_pointer.cpp
    INCLUDES tests
    LIBRARIES prosper_core)

  prosper_add_gtest(test_indirect_buffer_shadow
    SOURCES tests/gpu/recompiler/test_indirect_buffer_shadow.cpp
    LIBRARIES prosper_core)

    # Includes a shared fixture, which resolves from the `tests` root.
  prosper_add_gtest(test_exec_population_count
    SOURCES tests/gpu/test_exec_population_count.cpp
    INCLUDES tests
    LIBRARIES prosper_core)

  add_executable(test_indirect_pointer_static_footprint
    tests/gpu/recompiler/test_indirect_pointer_static_footprint.cpp)
    # Includes a shared fixture, which resolves from the `tests` root.
    target_include_directories(test_indirect_pointer_static_footprint PRIVATE tests)
  target_link_libraries(test_indirect_pointer_static_footprint PRIVATE prosper_core)
  add_test(NAME indirect_pointer_static_footprint
    COMMAND test_indirect_pointer_static_footprint)

  add_executable(test_indirect_pointer_descriptor_range
    tests/gpu/recompiler/test_indirect_pointer_descriptor_range.cpp)
    # Includes a shared fixture, which resolves from the `tests` root.
    target_include_directories(test_indirect_pointer_descriptor_range PRIVATE tests)
  target_link_libraries(test_indirect_pointer_descriptor_range PRIVATE prosper_core)
  add_test(NAME indirect_pointer_descriptor_range
    COMMAND test_indirect_pointer_descriptor_range)

  # Includes a shared fixture, which resolves from the `tests` root.
  prosper_add_gtest(test_gta5_cf9200
    SOURCES tests/gpu/recompiler/test_gta5_cf9200.cpp
    INCLUDES tests
    LIBRARIES prosper_core)

  prosper_add_gtest(test_compute_parent_walk
    SOURCES tests/gpu/diagnostics/test_compute_parent_walk.cpp
    LIBRARIES prosper_core)

  prosper_add_gtest(test_compute_tree_watch
    SOURCES tests/gpu/diagnostics/test_compute_tree_watch.cpp
    LIBRARIES prosper_core)

  # Cross-submit readability reuse is valid only while the HLE mapping generation is unchanged.
  prosper_add_gtest(test_guest_memory_map
    SOURCES tests/host/memory/test_guest_memory_map.cpp
    LIBRARIES prosper_core)

  prosper_add_gtest(test_renderer_tracked_mapping_cache
    SOURCES tests/hle/memory/test_renderer_tracked_mapping_cache.cpp
    INCLUDES frontends
    LIBRARIES prosper_core)

  add_executable(test_guest_memory_query tests/host/memory/test_guest_memory_query.cpp)
  target_link_libraries(test_guest_memory_query PRIVATE prosper_core)
  add_test(NAME guest_memory_query COMMAND test_guest_memory_query)

  # #3142: the chunked guest-memory content scan. The straddling case is the one that earns the
  # test -- a scanner that drops its overlap still returns a plausible hit list, so the blindness
  # is invisible in a live run and turns a "not found" into a wrong answer.
  prosper_add_gtest(test_guest_memory_search
    SOURCES tests/host/memory/test_guest_memory_search.cpp
    LIBRARIES prosper_core)

  # #3734: the single guest<->host copy policy every subsystem's helper forwards to -- never faults,
  # exact vs prefix, NULL / wrap / zero-length, errno preserved -- against a mapping with a hole in it.
  prosper_add_gtest(test_guest_memory_copy
    SOURCES tests/host/memory/test_guest_memory_copy.cpp
    LIBRARIES prosper_core)

  add_executable(test_guest_read_cache_disabled tests/host/memory/test_guest_read_cache_disabled.cpp)
  target_link_libraries(test_guest_read_cache_disabled PRIVATE prosper_core)
  add_test(NAME guest_read_cache_disabled COMMAND test_guest_read_cache_disabled)
  # #2387: guest_writable's range cache. The first case is the one that matters -- a read-only
  # mapping, deliberately warmed into the READABLE cache first, must still be refused for write.
  # A shared cache (or a writable path that consulted the readable registry) passes every other
  # assertion here and fails only that one, silently corrupting guest memory in production.
  # POSIX-only: the test builds its own read-only / read-write / revoked mappings with mmap and
  # mprotect, which is the whole point -- it needs to CREATE a page whose permissions it controls,
  # not find one. The Windows equivalent is VirtualAlloc + VirtualProtect(PAGE_READONLY) and is
  # deliberately left to the lane that can run it (docs/performance/THREAD_WAIT_PROFILING.md records the same
  # split for the profiling tools). The cache itself is built and exercised on all platforms; only
  # this test's mapping setup is POSIX.
  if(NOT WIN32)
    prosper_add_gtest(test_guest_writable_cache
      SOURCES tests/host/memory/test_guest_writable_cache.cpp
      LIBRARIES prosper_core)
  endif()

  # #2393: the caches above are guarded ONLY by host::guest_mapping_generation(), so every change
  # to guest page protection must advance it -- including the software watchpoints in the Linux
  # fault handler, which flip a live guest page to PROT_READ and back. Linux-only: it drives the
  # real arm path (prosper_arm_label_watch_force) against a page it maps itself. Both production
  # sources are handed to the wiring check, which confirms no protection change in the handler
  # bypasses the notifying wrapper and that the notifier is still lock-free.
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    add_executable(test_guest_writable_query_adapter
      tests/host/memory/test_guest_writable_query_adapter.cpp)
    target_link_libraries(test_guest_writable_query_adapter PRIVATE prosper_core)
    target_link_options(test_guest_writable_query_adapter PRIVATE "-Wl,--wrap=ioctl" "-Wl,--wrap=open")
    add_test(NAME guest_writable_query_adapter COMMAND ${CMAKE_COMMAND} -E env
      --unset=PROSPER_NO_GUEST_WRITE_CACHE --unset=PROSPER_NO_GUEST_WRITABLE_QUERY
      --unset=PROSPER_GUEST_WRITABLE_LOCAL_TEXT_CACHE $<TARGET_FILE:test_guest_writable_query_adapter>)
    foreach(scope_arm native text text-local)
      set(scope_env --unset=PROSPER_NO_GUEST_WRITE_CACHE
        --unset=PROSPER_NO_GUEST_WRITABLE_QUERY --unset=PROSPER_GUEST_WRITABLE_LOCAL_TEXT_CACHE)
      if(NOT scope_arm STREQUAL "native")
        list(APPEND scope_env PROSPER_NO_GUEST_WRITABLE_QUERY=1)
      endif()
      if(scope_arm STREQUAL "text-local")
        list(APPEND scope_env PROSPER_GUEST_WRITABLE_LOCAL_TEXT_CACHE=1)
      endif()
      add_test(NAME guest_writable_query_scope_${scope_arm} COMMAND ${CMAKE_COMMAND} -E env
        ${scope_env} $<TARGET_FILE:test_guest_writable_query_adapter> scope ${scope_arm})
    endforeach()

    add_executable(test_watch_mapping_generation tests/host/memory/test_watch_mapping_generation.cpp)
    target_link_libraries(test_watch_mapping_generation PRIVATE prosper_core)
    target_compile_definitions(test_watch_mapping_generation PRIVATE
      PROSPER_EXEC_IMAGE_LINUX_SOURCE="${CMAKE_CURRENT_SOURCE_DIR}/src/host/image/exec_image_linux.cpp"
      PROSPER_GUEST_MEMORY_MAP_SOURCE="${CMAKE_CURRENT_SOURCE_DIR}/src/host/memory/guest_memory_map.hpp"
      PROSPER_GUEST_WRITE_WATCH_SOURCE="${CMAKE_CURRENT_SOURCE_DIR}/src/host/memory/guest_write_watch.cpp")
    add_test(NAME watch_mapping_generation COMMAND test_watch_mapping_generation)
  endif()

  if(WIN32 OR CMAKE_SYSTEM_NAME STREQUAL "Linux")
    add_executable(test_guest_write_watch tests/host/memory/test_guest_write_watch.cpp)
    target_link_libraries(test_guest_write_watch PRIVATE prosper_core)
    add_test(NAME guest_write_watch
             COMMAND ${CMAKE_COMMAND} -E env --unset=PROSPER_WATCH_FORCE_PHYSICAL_SCAN
                     $<TARGET_FILE:test_guest_write_watch>)
    add_test(NAME guest_write_watch_forced_physical_scan
             COMMAND ${CMAKE_COMMAND} -E env PROSPER_WATCH_FORCE_PHYSICAL_SCAN=1
                     $<TARGET_FILE:test_guest_write_watch>)
    add_executable(test_guest_write_watch_index tests/host/memory/test_guest_write_watch_index.cpp)
    target_link_libraries(test_guest_write_watch_index PRIVATE prosper_core)
    # Two arms, and the environment is armed HERE rather than by a setenv inside main():
    # watch_query_audit_enabled() latches into a function-local static on its first call, so arming it
    # in-process is correct only while nobody moves a query above the setenv -- an ordering argument of
    # exactly the shape #2214 is about. The `off` arm is the one that covers the SHIPPED path; every
    # case except the audit's own positive control must hold identically in both.
    # The off arm UNSETS both switches rather than merely not setting them: a developer working on
    # this feature plausibly has them exported, and inheriting them turns the shipped-path arm into a
    # second audit arm (losing the only shipped-path coverage) or -- with PROSPER_WATCH_LEGACY_SCAN --
    # into a run where every assertion passes for the pre-#3681 reason. Neither announces itself.
    add_test(NAME guest_write_watch_index
             COMMAND ${CMAKE_COMMAND} -E env --unset=PROSPER_WATCH_QUERY_AUDIT
                     --unset=PROSPER_WATCH_LEGACY_SCAN
                     --unset=PROSPER_WATCH_FORCE_ALIAS_SCAN
                     $<TARGET_FILE:test_guest_write_watch_index>)
    add_test(NAME guest_write_watch_index_audit
             COMMAND ${CMAKE_COMMAND} -E env --unset=PROSPER_WATCH_LEGACY_SCAN
                     --unset=PROSPER_WATCH_FORCE_ALIAS_SCAN
                     PROSPER_WATCH_QUERY_AUDIT=1
                     $<TARGET_FILE:test_guest_write_watch_index>)
    # Third arm (#3681): the complete scan the index replaced. Until this existed the fallback had no
    # coverage at all -- both arms above UNSET the switch -- and, more importantly, nothing anywhere
    # asserted that the index REDUCES work. Every correctness case passes just as well against a
    # full scan, so an index that silently stopped narrowing would have given back the entire saving
    # while leaving the suite green. This arm is the negative control: identical dirty-tracking
    # assertions, opposite assertions about registrations visited.
    add_test(NAME guest_write_watch_index_legacy_scan
             COMMAND ${CMAKE_COMMAND} -E env --unset=PROSPER_WATCH_QUERY_AUDIT
                     --unset=PROSPER_WATCH_FORCE_ALIAS_SCAN
                     PROSPER_WATCH_LEGACY_SCAN=1
                     $<TARGET_FILE:test_guest_write_watch_index>)
    add_test(NAME guest_write_watch_index_forced_alias_scan
             COMMAND ${CMAKE_COMMAND} -E env --unset=PROSPER_WATCH_QUERY_AUDIT
                     --unset=PROSPER_WATCH_LEGACY_SCAN
                     PROSPER_WATCH_FORCE_ALIAS_SCAN=1
                     $<TARGET_FILE:test_guest_write_watch_index>)
  endif()
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    prosper_add_gtest(test_dmem_write_trace
      SOURCES tests/host/memory/test_dmem_write_trace.cpp
      LIBRARIES prosper_core)
  endif()

  # #1932: SIGTRAP arbitration between PROSPER_HWBP/PROSPER_HWWATCH and PROSPER_BP. Linux-only --
  # the live arm opens a real per-task perf execute breakpoint and patches a real int3 into this
  # test's own hand-assembled stubs. Links nothing: src/host/fault/trap_arbitration.hpp is header-only,
  # so the decision table also builds where the handler does not. The source path is handed to the
  # wiring check, which confirms the production handler still calls the shared predicate.
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    add_executable(test_trap_arbitration tests/host/fault/test_trap_arbitration.cpp)
    # Does not link prosper_core, so it does not inherit its PUBLIC `src`.
    target_include_directories(test_trap_arbitration PRIVATE src)
    target_compile_definitions(test_trap_arbitration PRIVATE
      PROSPER_EXEC_IMAGE_LINUX_SOURCE="${CMAKE_CURRENT_SOURCE_DIR}/src/host/image/exec_image_linux.cpp")
    add_test(NAME trap_arbitration COMMAND test_trap_arbitration)
  endif()

  # Unannounced-32-bit index-buffer fingerprint (#304 — DOLL's Slate/UMG quads). No Vulkan needed.
  prosper_add_gtest(test_index_size_detect
    SOURCES tests/gpu/execute/test_index_size_detect.cpp
    LIBRARIES prosper_core)

  # 16-bit index expansion: equivalence against the retained (uncalled) AVX2 kernel and the
  # PROT_NONE mapping edge. Header-only, links nothing, and needs no Vulkan — registered here
  # rather than beside the GPU-execution tests so a host without Vulkan still runs it.
  add_executable(test_index_expand tests/gpu/execute/test_index_expand.cpp)
  target_include_directories(test_index_expand PRIVATE src)
  add_test(NAME index_expand COMMAND test_index_expand)

  # Per-program compute dispatch facts are memoized by exact bytes and replay the probe's diagnostic
  # side effects; needs prosper_core but no Vulkan device.
  prosper_add_gtest(test_compute_program_facts
    SOURCES tests/gpu/execute/test_compute_program_facts.cpp
    LIBRARIES prosper_core)

  # Actual CPU draw/dispatch refusal producers: original versions, warm observation and cap evidence.
  prosper_add_gtest(test_refused_shader_producer
    SOURCES tests/gpu/execute/test_refused_shader_producer.cpp
    INCLUDES tests
    LIBRARIES prosper_core)

  # shader_histo: data-driven recompiler-coverage report over the game's real embedded RDNA2 shaders.
  add_executable(shader_histo tools/shader_histo/shader_histo.cpp)
  target_link_libraries(shader_histo PRIVATE prosper_core)

  # shader_inspect: bounded offline decode/CFG listing for raw PROSPER_SHADER_DUMP binaries.
  add_executable(shader_inspect tools/shader_inspect/shader_inspect.cpp)
  target_link_libraries(shader_inspect PRIVATE prosper_core)

  # Resource-table honesty (#1571): a raw dump carries no descriptors, so a table-less stage
  # rejection must be reported as an UNDETERMINED tool limitation, never as a shader defect.
  if(Python3_Interpreter_FOUND)
    add_test(NAME shader_inspect_resource_table_honesty
             COMMAND "${Python3_EXECUTABLE}"
                     "${CMAKE_CURRENT_SOURCE_DIR}/tools/shader_inspect/test_shader_inspect.py"
                     "$<TARGET_FILE:shader_inspect>")
  endif()

  # skip_survey's patterns are pinned to the EMITTER's format strings, and this is the
  # registration that makes that pinning real. Its worst failure is a pattern that silently
  # stops matching: every title then reports zero refused shaders, which is indistinguishable
  # from good news, on numbers already published (#3464). Instrument trap 272 is the same
  # shape and its own text is "the self-tests were registered; the format match was not" --
  # this test was the inverse until #3473 review, correct and running nowhere.
  #
  # Takes no target: it reads source files, so it needs no build artifact and no GPU.
  if(Python3_Interpreter_FOUND)
    add_test(NAME skip_survey_patterns_match_emitters
             COMMAND "${Python3_EXECUTABLE}"
                     "${CMAKE_CURRENT_SOURCE_DIR}/tools/shader_inspect/test_skip_survey.py")
  endif()

  # The baseline comparison's non-obvious verdicts: a run that cannot be judged must FAIL, and a
  # different GPU must refuse to compare at all. Both are ways the guard could go green while
  # measuring nothing -- an AMD host refuses no shaders, so an NVIDIA baseline compared there
  # would report every shader fixed. Pure logic over fixtures: no dump, no GPU, no boot.
  if(Python3_Interpreter_FOUND)
    add_test(NAME skip_baseline_refuses_what_it_cannot_compare
             COMMAND "${Python3_EXECUTABLE}"
                     "${CMAKE_CURRENT_SOURCE_DIR}/tools/shader_inspect/test_skip_baseline.py")
  endif()

  # Whether a fragment module answers the same over two 32-lane groups as over one 64-lane wave --
  # the property the wave32 allowance actually needs, and NOT the one the shipping classifier tests
  # (it tests the reason set, i.e. how the vote was produced). Its fixtures are hand-assembled
  # SPIR-V, so each case exists independently of what the emitter happens to produce.
  #
  # Half the cases must be REFUSED, including the one that retired the earlier value-reachability
  # predicate: a vote used only as a branch condition never flows into a store, yet it decides which
  # store runs. The divergent and uniform arms are the same three shapes with one word changed, so
  # a uniform arm that did nothing and one that did too much both fail rather than look perfect.
  if(Python3_Interpreter_FOUND)
    add_test(NAME wave_width_independence_refuses_the_class
             COMMAND "${Python3_EXECUTABLE}"
                     "${CMAKE_CURRENT_SOURCE_DIR}/tools/shader_inspect/test_wave_width_independent.py"
                     "$<TARGET_FILE:test_wave_width_independent>")
  endif()

  # spv_validate: emit a module from EVERY SPIR-V-producing entry point and run spirv-val on each —
  # a strict-validation gate (the render tests only prove llvmpipe accepts them; llvmpipe is
  # lenient). No Vulkan needed, but spirv-val (spirv-tools) IS required: it used to report PASS when
  # the validator was absent, which is how #1711 shipped invalid SPIR-V past a gate CLAUDE.md and
  # both READMEs describe as strict. The source root is passed so the tool can read the emitter
  # headers and fail when a new emitter is added without a module here.
  add_executable(spv_validate tools/spv_validate/spv_validate.cpp)
  target_link_libraries(spv_validate PRIVATE prosper_core)
  add_test(NAME spv_validate
           COMMAND spv_validate ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

  # tools/vkprobe/shaders (#2945): the HAND-WRITTEN SPIR-V controls -- the arm that separated
  # "prosper's generated SPIR-V is malformed" from "the driver is wrong", by performing the same
  # storage-buffer loads from a module prosper did not emit. Their entire value is being valid by
  # construction, and this folder's recorded history is a control that reported confident numbers
  # from an INVALID pipeline for weeks because nothing checked it. So the check is a test, not a
  # README line: every .spvasm must assemble, land on SPIR-V 1.3, and pass spirv-val.
  #
  # Skipped, loudly, when the assembler is absent -- a validation gate that passes because its
  # validator is missing is exactly the #1711 failure spv_validate above exists to prevent, and a
  # silent skip is how instrument trap 126 reads a smaller suite as a green one.
  find_program(SPIRV_AS_EXECUTABLE NAMES spirv-as)
  find_program(SPIRV_VAL_EXECUTABLE NAMES spirv-val)
  if(Python3_Interpreter_FOUND AND SPIRV_VAL_EXECUTABLE)
    add_test(NAME storage_write_mask_spirv_validate
             COMMAND "${Python3_EXECUTABLE}"
                     "${CMAKE_SOURCE_DIR}/frontends/shared/tests/test_storage_write_mask_spirv.py"
                     $<TARGET_FILE:test_storage_write_mask_spirv> "${SPIRV_VAL_EXECUTABLE}")
  else()
    message(STATUS "storage_write_mask_spirv_validate NOT registered: Python3 or spirv-val missing")
  endif()
  if(Python3_Interpreter_FOUND AND SPIRV_AS_EXECUTABLE AND SPIRV_VAL_EXECUTABLE)
    add_test(NAME vkprobe_hand_written_controls
             COMMAND "${Python3_EXECUTABLE}"
                     "${CMAKE_CURRENT_SOURCE_DIR}/tools/vkprobe/shaders/check_controls.py"
                     "${SPIRV_AS_EXECUTABLE}" "${SPIRV_VAL_EXECUTABLE}"
                     "${CMAKE_CURRENT_SOURCE_DIR}/tools/vkprobe/shaders"
                     "${CMAKE_CURRENT_BINARY_DIR}")
  else()
    message(STATUS "vkprobe_hand_written_controls NOT registered: "
                   "python3=${Python3_Interpreter_FOUND} spirv-as=${SPIRV_AS_EXECUTABLE} "
                   "spirv-val=${SPIRV_VAL_EXECUTABLE}")
  endif()

  # #2945: a recompiled VS must not declare more output components than the device can carry, and
  # the bound is what the FS actually reads. Carries its own mutation arm. Registered OUTSIDE the
  # Vulkan_FOUND block on purpose -- it inspects SPIR-V words and needs no device, so gating it on
  # Vulkan would silently delete the guard on a configuration without it.
  prosper_add_gtest(test_vertex_output_budget
    SOURCES tests/gpu/recompiler/test_vertex_output_budget.cpp
    INCLUDES tests
    LIBRARIES prosper_core)

  # AGC→Vulkan resource layer: the front-half↔back-half contract (gpu_resources.hpp). Registry
  # behavior is fixed here so both halves integrate against a stable API. Pure/cross-platform.
  prosper_add_gtest(test_gpu_resources
    SOURCES tests/gpu/resources/test_gpu_resources.cpp
    LIBRARIES prosper_core)

  # Resource-binding contract (shader_resources.hpp): format sizing + the provenance/binding lookups
  # the recompiler and pipeline share. Pure/cross-platform. See docs/gpu/RESOURCE_BINDING.md.
  # Spells a `shared/...` include, which resolves from `frontends`.
  prosper_add_gtest(test_shader_resources
    SOURCES tests/gpu/resources/test_shader_resources.cpp
    INCLUDES frontends
    LIBRARIES prosper_core)

  # Versioned realized-draw capture: explicit round-trip, overlapping-resource alias preservation,
  # owned replay backing, and corrupt/truncated input rejection. Pure; no Vulkan or game dump.
  add_executable(test_gpu_capture tests/gpu/capture/test_gpu_capture.cpp)
  # Includes a shared fixture, which resolves from the `tests` root.
  target_include_directories(test_gpu_capture PRIVATE tests)
  target_link_libraries(test_gpu_capture PRIVATE prosper_core)
  add_test(NAME gpu_capture_roundtrip COMMAND test_gpu_capture)
  prosper_add_gtest(test_gpu_capture_observation
    SOURCES tests/gpu/capture/test_gpu_capture_observation.cpp
    INCLUDES tests
    LIBRARIES prosper_core)

  # Actual CPU draw realization -> capture -> replay retains guest width independently of SPIR-V.
  prosper_add_gtest(test_realized_fragment_width
    SOURCES tests/gpu/execute/test_realized_fragment_width.cpp
    LIBRARIES prosper_core)
  add_executable(test_fragment_float_mode_capture tests/gpu/execute/test_fragment_float_mode_capture.cpp)
  target_link_libraries(test_fragment_float_mode_capture PRIVATE prosper_core)
  add_test(NAME fragment_float_mode_capture COMMAND test_fragment_float_mode_capture)
  add_executable(test_fragment_float_flags tests/gpu/state/test_fragment_float_flags.cpp)
  target_link_libraries(test_fragment_float_flags PRIVATE prosper_core)
  add_test(NAME fragment_float_flags COMMAND test_fragment_float_flags)
  # #3549: a storage-image or format-fetch DST_SEL is part of the compiled-shader key.
  prosper_add_gtest(test_shader_key_dst_sel
    SOURCES tests/gpu/execute/test_shader_key_dst_sel.cpp
    INCLUDES src
    LIBRARIES prosper_core)
  add_executable(test_fragment_float_flags_key tests/gpu/execute/test_fragment_float_flags_key.cpp)
  target_link_libraries(test_fragment_float_flags_key PRIVATE prosper_core)
  add_test(NAME fragment_float_flags_key COMMAND test_fragment_float_flags_key)
  add_executable(test_float_transport_capture tests/gpu/execute/test_float_transport_capture.cpp)
  target_link_libraries(test_float_transport_capture PRIVATE prosper_core)
  add_test(NAME float_transport_capture COMMAND test_float_transport_capture)
  add_test(NAME float_transport_null_export_explicit_first
    COMMAND test_float_transport_capture --dcc-explicit-first)

  # Includes a shared fixture, which resolves from the `tests` root.
  prosper_add_gtest(test_gpu_capture_bundle
    SOURCES tests/gpu/capture/test_gpu_capture_bundle.cpp
    INCLUDES tests
    LIBRARIES prosper_core)

  add_executable(test_gpu_capture_bundle_streaming tests/gpu/capture/test_gpu_capture_bundle_streaming.cpp)
  target_include_directories(test_gpu_capture_bundle_streaming PRIVATE tests)
  target_link_libraries(test_gpu_capture_bundle_streaming PRIVATE prosper_core)
  add_test(NAME gpu_capture_bundle_streaming COMMAND test_gpu_capture_bundle_streaming)
  set_tests_properties(gpu_capture_bundle_streaming PROPERTIES TIMEOUT 60)

  add_executable(test_interactive_capture_writer tests/gpu/capture/test_interactive_capture_writer.cpp)
  target_include_directories(test_interactive_capture_writer PRIVATE tests)
  target_link_libraries(test_interactive_capture_writer PRIVATE prosper_core)
  add_test(NAME interactive_capture_writer COMMAND test_interactive_capture_writer)
  add_test(NAME interactive_capture_writer_shutdown_collecting COMMAND test_interactive_capture_writer shutdown)
  set_tests_properties(interactive_capture_writer interactive_capture_writer_shutdown_collecting
                       PROPERTIES TIMEOUT 30)

  # Framerate must count DISTINCT guest frames, not publications: the renderer re-serves its retained
  # frame through the ordinary publish path, so a present-counting fps reads full speed for a frozen
  # title (#2783 / instrument trap 90). Pure -- no Vulkan, no dump.
  prosper_add_gtest(test_present_frame_rate
    SOURCES tests/gpu/present/test_present_frame_rate.cpp
    LIBRARIES prosper_core)

  # Native-speed append-only submit/present timeline: portable framing, checksums, and truncated-tail
  # recovery are pure and must not depend on Vulkan or a game dump.
  add_executable(test_gpu_timeline tests/gpu/timeline/test_gpu_timeline.cpp)
  # Includes a shared fixture, which resolves from the `tests` root.
  target_include_directories(test_gpu_timeline PRIVATE tests)
  target_link_libraries(test_gpu_timeline PRIVATE prosper_core)
  add_test(NAME gpu_timeline_roundtrip COMMAND test_gpu_timeline)

  prosper_add_gtest(test_menu_frame_gate
    SOURCES tests/gpu/timeline/test_menu_frame_gate.cpp
    LIBRARIES prosper_core)

  add_executable(test_gpu_timeline_capture_exit tests/gpu/test_gpu_timeline_capture_exit.cpp)
  # Includes a shared fixture, which resolves from the `tests` root.
  target_include_directories(test_gpu_timeline_capture_exit PRIVATE tests)
  target_link_libraries(test_gpu_timeline_capture_exit PRIVATE prosper_core)
  add_test(NAME gpu_timeline_capture_exit COMMAND test_gpu_timeline_capture_exit)

  # Generic guest-log phase gate: exact bounded lines arm the existing seeded whole-frame capture
  # after one completed present. Pure/cross-platform; no game dump or Vulkan device.
  # Includes a shared fixture, which resolves from the `tests` root.
  prosper_add_gtest(test_guest_log_capture
    SOURCES tests/gpu/test_guest_log_capture.cpp
    INCLUDES tests
    LIBRARIES prosper_core)

  # The same gate's SILENT failure: an armed marker that never matches must diagnose itself instead
  # of leaving a missing file as the only evidence (#1684). Separate binary because the gate is a
  # process-wide one-shot and guest_log_capture's marker deliberately fires.
  add_executable(test_guest_log_capture_miss tests/gpu/timeline/test_guest_log_capture_miss.cpp)
  # Includes a shared fixture, which resolves from the `tests` root.
  target_include_directories(test_guest_log_capture_miss PRIVATE tests)
  target_link_libraries(test_guest_log_capture_miss PRIVATE prosper_core)
  add_test(NAME guest_log_capture_miss COMMAND test_guest_log_capture_miss)

  # The same silence, two selector families over: a malformed capture tunable substituting a
  # different policy (#2565) and a detailed timeline selector that never matched (#2564). Separate
  # binary because both the detailed-capture request and the tunable refusal are process-wide
  # one-shots, and one of its arms deliberately ends a CHILD process during static initialization.
  add_executable(test_capture_tunables tests/gpu/timeline/test_capture_tunables.cpp)
  # Includes a shared fixture, which resolves from the `tests` root.
  target_include_directories(test_capture_tunables PRIVATE tests)
  target_link_libraries(test_capture_tunables PRIVATE prosper_core)
  add_test(NAME capture_tunables COMMAND test_capture_tunables)

  # Remote/headless F9 equivalent: an external regular-file witness arms the existing seeded
  # whole-frame capture on the next present. Pure/cross-platform; no game dump or Vulkan device.
  add_executable(test_capture_trigger_file tests/gpu/test_capture_trigger_file.cpp)
  # Includes a shared fixture, which resolves from the `tests` root.
  target_include_directories(test_capture_trigger_file PRIVATE tests)
  target_link_libraries(test_capture_trigger_file PRIVATE prosper_core)
  add_test(NAME capture_trigger_file COMMAND test_capture_trigger_file)

  # gpu_replay output dimensions follow the selected draw/prefix target when its byte count agrees.
  # Pure/cross-platform so focused replay tooling remains available in CI.
  prosper_add_gtest(test_gpu_replay_extent
    SOURCES tests/test_gpu_replay_extent.cpp
    LIBRARIES prosper_core)

  add_executable(test_gpu_replay_shader_dump tests/gpu/diagnostics/test_gpu_replay_shader_dump.cpp)
  target_link_libraries(test_gpu_replay_shader_dump PRIVATE prosper_core)
  add_test(NAME gpu_replay_shader_dump COMMAND test_gpu_replay_shader_dump)

  add_executable(test_replay_fragment_width tests/gpu/recompiler/test_replay_fragment_width.cpp)
  target_link_libraries(test_replay_fragment_width PRIVATE prosper_core)
  add_test(NAME replay_fragment_width COMMAND test_replay_fragment_width)

  # Draw-scoped captured-resource replacement: strict semantic selector, exact span validation,
  # and copy-on-write isolation when draws share tables/backing. Pure/cross-platform; no Vulkan.
                 # Includes a shared fixture, which resolves from the `tests` root.
  prosper_add_gtest(test_gpu_replay_resource_override
    SOURCES tests/test_gpu_replay_resource_override.cpp
    INCLUDES tests
    LIBRARIES prosper_core)

  # Selected-prefix compute output inspection must distinguish captured seed bytes from the
  # descriptor-visible post-dispatch image, including tiled 2D and true 3D storage layouts.
  prosper_add_gtest(test_gpu_replay_post_compute_resource
    SOURCES tests/test_gpu_replay_post_compute_resource.cpp
    LIBRARIES prosper_core)

  # Offline SPI_PS_INPUT_CNTL linkage classification: a pixel-shader input wired to the constant
  # form consumes no producer export, and must never be reported as a PARAM consumer.
  prosper_add_gtest(test_gpu_replay_pixel_inputs
    SOURCES tests/test_gpu_replay_pixel_inputs.cpp
    LIBRARIES prosper_core)

  # Compile-only synthetic reject-route guard for #2790; not an executable real-shader reproducer.
  prosper_add_gtest(test_undef_loop_accumulator
    SOURCES tests/gpu/recompiler/test_undef_loop_accumulator.cpp
    LIBRARIES prosper_core)

  # Fail-visible refusal pins for gap opcodes (v_perm_b32, v_mad_i64_i32,
  # v_div_fixup_f32, v_sad_u8, s_movrels_b32, image_gather4): empty module,
  # never a silent mistranslation. Compile-only, no device. Any later
  # lowering replaces these arms with execution tests.
  prosper_add_gtest(test_gap_opcode_refusals
    SOURCES tests/gpu/recompiler/test_gap_opcode_refusals.cpp
    LIBRARIES prosper_core)

  add_executable(test_fragment_loop_mask tests/gpu/recompiler/test_fragment_loop_mask.cpp)
  target_link_libraries(test_fragment_loop_mask PRIVATE prosper_core)
  add_test(NAME fragment_loop_mask COMMAND test_fragment_loop_mask)

  # Native-typed readback formatting: the property asserted is the one an RGBA8 conversion
  # destroys, so these arms fail if a scalar is ever routed through one again (#3765).
  prosper_add_gtest(test_gpu_replay_native_texels
    SOURCES tests/test_gpu_replay_native_texels.cpp
    LIBRARIES prosper_core)

  # The RGBA8 picture of a seed must cover every GpuCaptureColorFormat member (#3765: five were
  # dropped to an empty image). -Werror=switch here as on gpu_replay, so the header's exhaustive
  # switches fail the build for a new enumerator even in a configuration that skips gpu_replay.
  add_executable(test_gpu_replay_rtt_seed_rgba8 tests/test_gpu_replay_rtt_seed_rgba8.cpp)
  target_link_libraries(test_gpu_replay_rtt_seed_rgba8 PRIVATE prosper_core)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(test_gpu_replay_rtt_seed_rgba8 PRIVATE -Werror=switch)
  endif()
  add_test(NAME gpu_replay_rtt_seed_rgba8 COMMAND test_gpu_replay_rtt_seed_rgba8)

  prosper_add_gtest(test_gpu_dependency_graph
    SOURCES tests/gpu/execute/test_gpu_dependency_graph.cpp
    LIBRARIES prosper_core)

  add_executable(gpu_timeline tools/gpu_timeline/gpu_timeline.cpp)
  target_link_libraries(gpu_timeline PRIVATE prosper_core)

  # Front-half resource-table builder + V# descriptor decode (agc_shader_layout.cpp): shader user_data
  # + user-data SGPRs -> ShaderResourceTable (constant buffers, stage 1). Pure/cross-platform.
  prosper_add_gtest(test_build_shader_resources
    SOURCES tests/gpu/agc/test_build_shader_resources.cpp
    LIBRARIES prosper_core)

  # GPU descriptor-bound INVARIANT guard (agc_shader_layout.cpp decode_image/buffer_descriptor): the
  # entire guest-input memory-safety argument relies on width/height <= 16384, depth <= 8192, and V#
  # size_bytes clamped to 32 bits. Pins those exact ceilings so a mask-widening / +1-bias refactor fails
  # here first instead of silently un-bounding every downstream size product. Pure decode; no dump/Vulkan.
  prosper_add_gtest(test_descriptor_clamp
    SOURCES tests/gpu/agc/test_descriptor_clamp.cpp
    LIBRARIES prosper_core)

  # #3205: mip-chain provenance participates in image IDENTITY only when a chain exists. The fields
  # are populated on one construction path and left unset on another, so comparing them
  # unconditionally split identical single-level descriptors into separate Vulkan images -- which
  # dropped three of four quadrant stores in GTA V and rendered a checkerboard. Pure predicate.
  prosper_add_gtest(test_mip_provenance_identity
    SOURCES tests/gpu/agc/test_mip_provenance_identity.cpp
    LIBRARIES prosper_core)

  # #3204: the image-identity predicates extracted from execute_item. They decide whether two
  # bindings alias ONE Vulkan image -- the aliasing GTA V's four quadrant writes depend on, and
  # the thing #3205 broke while it was an unnamed local expression. Pure comparisons.
  prosper_add_gtest(test_image_identity
    SOURCES tests/gpu/agc/test_image_identity.cpp
    LIBRARIES prosper_core)

  # #3195: the atomic-image LOGICAL/PHYSICAL staging extents, extracted from execute_item.
  prosper_add_gtest(test_atomic_image_staging
    SOURCES tests/gpu/agc/test_atomic_image_staging.cpp
    LIBRARIES prosper_core)

  # #3204: SPIR-V/guest storage-format agreement, extracted from execute_item. The permissive
  # direction binds the guest image with the wrong element type, so every arm has a rejection
  # partner -- including atomics, which are never a native storage image however well formats match.
  prosper_add_gtest(test_spirv_storage_match
    SOURCES tests/gpu/agc/test_spirv_storage_match.cpp
    LIBRARIES prosper_core)

  # Descriptor-binding policy (#157): textures/storage images never land on the recompiler's hardwired
  # cbuf bindings 2/3; constant/vertex buffers assigned first. Pure, no game dump / Vulkan needed.
  prosper_add_gtest(test_convention_bindings
    SOURCES tests/gpu/test_convention_bindings.cpp
    LIBRARIES prosper_core)

  add_executable(test_writer_provenance tests/gpu/capture/test_writer_provenance.cpp)
  target_link_libraries(test_writer_provenance PRIVATE prosper_core)
  add_test(NAME writer_provenance_ranges COMMAND test_writer_provenance)
  # The provenance switch is sampled once per process. Arm it before launch so this fixture
  # cannot pass with a stale cached value from an earlier in-process read.
  set_tests_properties(writer_provenance_ranges PROPERTIES
                       ENVIRONMENT "PROSPER_PROVENANCE_DIM=3200x1800")

  # The init-fault report must survive a wild/null faulting RIP (#128). On Windows this also drives
  # the VEH recovery thunk and asserts its compiled-helper call obeys the Microsoft-x64 ABI (#633).
  add_executable(test_initfault_dump tests/host/image/test_initfault_dump.cpp)
  target_link_libraries(test_initfault_dump PRIVATE prosper_core)
  add_test(NAME initfault_dump_survives COMMAND test_initfault_dump)

  if(WIN32)
    prosper_add_gtest(test_win_recovery
      SOURCES tests/host/image/test_win_recovery.cpp tests/host/image/win_recovery_witness.cpp
      LIBRARIES prosper_core
      PROPERTIES TIMEOUT 15)
  endif()

  # Generated guest->host import boundary: preserve all register args, stack args 7-9, return value,
  # and ABI stack alignment. Windows exercises SysV->Microsoft-x64; Linux also drives the FS-swap path.
  add_executable(test_hle_stack_args tests/host/image/test_hle_stack_args.cpp)
  target_link_libraries(test_hle_stack_args PRIVATE prosper_core)
  add_test(NAME hle_stack_args COMMAND test_hle_stack_args)
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    add_test(NAME hle_stack_args_guest_fs COMMAND test_hle_stack_args --guest-fs)
  endif()

  # #2955: the guest(SysV) -> host(Microsoft x64) import bridge. CROSS-PLATFORM on purpose, and that
  # is the point rather than an accident: the bridge's placement rules are a pure function of a
  # handler's signature, so they can be asserted anywhere, and on any x86-64 host the emitted bytes
  # can additionally be EXECUTED across the SysV/Microsoft boundary via __attribute__((ms_abi)).
  # Gating this on WIN32 would leave the one defect it exists to catch visible only on the platform
  # nobody develops on. Its non-x86-64 arm still checks the tables.
  prosper_add_gtest(test_sysv_ms_bridge
    SOURCES tests/host/abi/test_sysv_ms_bridge.cpp
    LIBRARIES prosper_core)
  add_executable(test_guest_tailjump tests/host/abi/test_guest_tailjump.cpp)
  target_link_libraries(test_guest_tailjump PRIVATE prosper_core)
  add_test(NAME guest_tailjump COMMAND test_guest_tailjump)
  set_tests_properties(guest_tailjump PROPERTIES SKIP_RETURN_CODE 77)

  # #3246: the one argument list #2955's CallSignature cannot describe — a real C variadic, whose
  # shape the format string decides at run time. CROSS-PLATFORM for the same reason as the bridge
  # above, and then some: reading a guest System V va_list and writing a Microsoft one are pure
  # functions, and BOTH conventions' va_arg are available on any x86-64 host through
  # __builtin_sysv_va_list / __builtin_ms_va_list — so the round trip is executed against the
  # compiler's own readers here, and against the real CRT on Windows.
  add_executable(test_guest_varargs tests/host/abi/test_guest_varargs.cpp)
  target_link_libraries(test_guest_varargs PRIVATE prosper_core)
  prosper_add_gnu_variadic_producer(test_guest_varargs
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/host/abi/guest_varargs_fixture.cpp" test_capture)
  add_test(NAME guest_varargs COMMAND test_guest_varargs)

  # sceImeUpdate must invoke the guest key handler on the CALLER's guest %fs, not the host %fs the
  # import stub swapped in (#1286: Evergate SIGSEGV at eboot+0xaf8431 on a key press). Drives the real
  # import path on an activated guest TCB; a no-op pass off Linux.
  prosper_add_gtest(test_ime_fs
    SOURCES tests/hle/input/test_ime_fs.cpp
    LIBRARIES prosper_core)

  # x86-64 low-address READ decoder used by the Windows PROSPER_NULL_PAGE emulation. Header-only, so the
  # decode logic is verified on every platform even though the VEH caller is Windows-only.
  prosper_add_gtest(test_x86_read_decode
    SOURCES tests/host/x86/test_x86_read_decode.cpp
    INCLUDES src)

  # #2018: a fatal worker-thread fault report must describe one fault. Header-only snapshot of the
  # handler's own siginfo/ucontext; POSIX signal contexts only (the test self-skips elsewhere).
  # Does not link prosper_core, so it does not inherit its PUBLIC `src`.
  # The contention arm spawns real threads. This target links no library, so unlike every other
  # thread-using test here it inherits -pthread from nothing; name Threads::Threads explicitly
  # (found above with prosper_core, whose existence gates this block).
  prosper_add_gtest(test_fault_context
    SOURCES tests/host/fault/test_fault_context.cpp
    INCLUDES src
    LIBRARIES Threads::Threads)

  # End-to-end AGC submit plus fixed args 7-9 in ReleaseMem/WaitRegMem. This is cross-platform so
  # Windows validates the real fence builders behind the generated import-ABI conformance test.
  prosper_add_gtest(test_agc_dcb_callback_abi
    SOURCES tests/hle/test_agc_dcb_callback_abi.cpp
    LIBRARIES prosper_core)

  prosper_add_gtest(test_agc_submit
    SOURCES tests/gpu/pm4/test_agc_submit.cpp
    LIBRARIES prosper_core)

  # sceAgcDriverSubmitMultiDcbs: zero-length segments are skipped, not fatal.
  prosper_add_gtest(test_agc_multi_dcb_empty_segment
    SOURCES tests/gpu/pm4/test_agc_multi_dcb_empty_segment.cpp
    LIBRARIES prosper_core)

  add_executable(test_agc_submit_multi tests/hle/test_agc_submit_multi.cpp)
  target_include_directories(test_agc_submit_multi PRIVATE tests)
  target_link_libraries(test_agc_submit_multi PRIVATE prosper_core)
  add_test(NAME agc_submit_multi COMMAND test_agc_submit_multi)

  # libSceVideoOut: query/config functions return real 1080p60 values + swapchain-scaffolding
  # display-buffer registry records the game's registered framebuffers.
  #
  # CROSS-PLATFORM on purpose (#1765). This sat inside the if(UNIX) block below, so the one platform
  # whose display pacing is at risk was the one platform that did not build it: sceVideoOutWaitVblank
  # waits on a deadline, and on Windows that wait is quantized by the process timer period. Nothing
  # here is POSIX-specific -- it drives the HLE registry and asserts returned values -- so the gate
  # was incidental rather than a statement about the test.
  prosper_add_gtest(test_videoout
    SOURCES tests/host/platform/test_videoout.cpp
    LIBRARIES prosper_core)

  # #3017: the display prosper advertises must be DERIVED from the host's real mode rather than
  # hardcoded 1080p59.94. Its own executable, not an arm of test_videoout, because the advertised
  # mode resolves ONCE per process on first use -- so the derived case needs a process whose
  # environment is set before any VideoOut call, and test_videoout above is exactly the
  # complementary process where those variables are unset and the untouched `legacy` default is
  # asserted. CROSS-PLATFORM for the same reason test_videoout is: it drives the HLE registry and
  # a pure header, with no OS display access.
  prosper_add_gtest(test_display_mode
    SOURCES tests/hle/graphics/test_display_mode.cpp
    LIBRARIES prosper_core)

  # #3379: guest flips are paced at the cadence the TITLE asked for by default. Its own executable
  # rather than an arm of test_videoout, for the same reason test_display_mode above is: the
  # advertised display mode resolves once per process, and these arms read the derived period
  # against the untouched `legacy` default. CROSS-PLATFORM like both of those -- HLE registry plus
  # a sleep, no display access. Includes one deliberately one-sided timing arm (a lower bound on a
  # sleep cannot flake).
  prosper_add_gtest(test_flip_pacing
    SOURCES tests/host/platform/test_flip_pacing.cpp
    LIBRARIES prosper_core)

  # precise_sleep's deadline post-condition (#3074): every sleep_until_steady_ns() backend must
  # never return before now_ns() >= deadline_ns. Win32SleepFallback did not re-check after its one
  # OS sleep call, unlike the sibling high-resolution-timer path. CROSS-PLATFORM like test_videoout
  # above and for the same reason: sleep_until_deadline_retry() is the pure-function loop the fix
  # composes, deliberately extracted so its termination condition is testable with a fake clock and
  # a fake sleep on any host, without executing the Windows-only code path or measuring wall time.
  prosper_add_gtest(test_precise_sleep
    SOURCES tests/host/platform/test_precise_sleep.cpp
    LIBRARIES prosper_core)

  # Windows guest-execution substrate (exec_image_win.cpp): build boot_trace so the whole boot path
  # links (proves the substrate + SysV-ABI HLE boundary resolve). No evdev/Vulkan here. The guest boot
  # itself is not yet verified on Windows — see docs/platforms/PORTING.md "Windows".
  if(WIN32)
    add_executable(boot_trace tools/boot_trace/boot_trace.cpp tools/boot_trace/first_frame_capture.cpp)
    # Includes a shared fixture, which resolves from the `tests` root.
    target_include_directories(boot_trace PRIVATE tests)
    # Spells a `shared/...` include, which resolves from `frontends`.
    target_include_directories(boot_trace PRIVATE frontends)
    target_link_libraries(boot_trace PRIVATE prosper_core)

    # Live Vulkan renderer on Windows (PROSPER_RENDER=1): with the Vulkan SDK present, wire the same
    # shared DrawItem->Vulkan compositor boot_trace uses on Linux so the guest's SubmitDcb draws
    # execute + present + dump frames. Mirrors the UNIX block below. `gpu_replay` is built here too
    # since #2331: the F9/scheduled bundle grab already worked on Windows, so a Windows investigation
    # could produce a several-hundred-MiB .prgbundle and then have no way to open it -- the documented
    # "capture, then replay offline" loop was half-available, and the half that was missing was the
    # half that answers the question. See docs/platforms/PORTING.md "Windows".
    if(NOT TARGET Vulkan::Vulkan)
      find_package(Vulkan QUIET)
    endif()
    if(Vulkan_FOUND)
      target_link_libraries(boot_trace PRIVATE Vulkan::Vulkan)
      target_compile_definitions(boot_trace PRIVATE PROSPER_HAVE_VULKAN=1)
      add_library(prosper_live_renderer STATIC frontends/shared/live/live_renderer.cpp
                                               frontends/shared/live/submit_renderer/callback_state.cpp
                                               frontends/shared/live/submit_renderer/callback_thread_state.cpp
                                               frontends/shared/live/submit_renderer/draw_resources.cpp
                                               frontends/shared/live/submit_renderer/image_resources.cpp
                                               frontends/shared/live/submit_renderer/timing_report.cpp
                                               frontends/shared/live/submit_renderer/backend_draws.cpp
                                               frontends/shared/live/submit_renderer/backend_timing.cpp
                                               frontends/shared/live/submit_renderer/per_target_passes.cpp
                                               frontends/shared/live/submit_renderer/resolve_pass.cpp
                                               frontends/shared/live/submit_renderer/pass_diagnostics.cpp
                                               frontends/shared/live/submit_renderer/callback_prelude.cpp
                                               frontends/shared/live/submit_renderer/callback_diagnostics.cpp
                                               frontends/shared/live/submit_renderer/single_framebuffer.cpp
                                               frontends/shared/live/submit_renderer/final_span_present.cpp
                                               frontends/shared/live/live_compute.cpp
                                               frontends/shared/present/present_blit.cpp
                                               frontends/shared/present/compute_scanout.cpp)
      target_include_directories(prosper_live_renderer PRIVATE src tests tools/boot_trace)
      target_link_libraries(prosper_live_renderer PUBLIC prosper_core Vulkan::Vulkan
                                                   prosper_performance_capture)
      target_include_directories(prosper_live_renderer PUBLIC frontends)
      # A LiveTargetPixelFormat mapping that quietly falls back to RGBA8 has now cost two titles a
      # whole render layer (#773 Dead Cells, and Syberia's black 3D menu). frontends/shared/
      # live_target_format.hpp maps the enum with default-less switches; -Werror=switch promotes a
      # member added without updating them from a silent wrong format to a build failure.
      if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(prosper_live_renderer PRIVATE -Werror=switch)
      endif()

      # #3204: the renderer-target format table, extracted from execute_item where it was an
      # `||` chain -- the shape that cost Dead Cells an HDR target (#773) and Syberia its 3D
      # menu. Now an exhaustive switch: the compiler pins exhaustiveness, this test pins the
      # table. Registered HERE, inside if(Vulkan_FOUND), because live_target_format.hpp includes
      # <vulkan/vulkan.h> -- registering it outside built on Linux and broke MinGW and macOS.
      prosper_add_gtest(test_live_target_format_match
        SOURCES tests/gpu/test_live_target_format_match.cpp
        INCLUDES frontends src
        LIBRARIES prosper_core Vulkan::Vulkan)
      target_link_libraries(boot_trace PRIVATE prosper_live_renderer)

      # Keep the production compute backend covered on native Windows too. This catches host- and
      # driver-specific image upload/writeback failures in the same backend prosper-app uses.
      add_executable(test_game_compute tests/gpu/recompiler/test_game_compute.cpp)
      # Includes a shared fixture, which resolves from the `tests` root.
      target_include_directories(test_game_compute PRIVATE tests)
      # Spells a `shared/...` include, which resolves from `frontends`.
      target_include_directories(test_game_compute PRIVATE frontends)
      target_link_libraries(test_game_compute PRIVATE prosper_live_renderer)
      add_executable(test_game_compute_rtt_mirror tests/gpu/recompiler/test_game_compute_rtt_mirror.cpp)
      target_include_directories(test_game_compute_rtt_mirror PRIVATE tests frontends)
      target_link_libraries(test_game_compute_rtt_mirror PRIVATE prosper_live_renderer)
      # #3656: executor half of device-resolved indirect dispatch (recording backend, no Vulkan work).
      prosper_add_gtest(test_indirect_dispatch_device_route
        SOURCES tests/gpu/execute/test_indirect_dispatch_device_route.cpp
        LIBRARIES prosper_core)
      # #3656: the Vulkan half -- the production backend consuming a device-resolved dispatch with
      # vkCmdDispatchIndirect from a retained buffer. Registered for native Windows because its
      # discriminating arm (a stale guest triplet) is driver-sensitive.
      add_executable(test_live_compute_indirect_dispatch
                     tests/gpu/execute/test_live_compute_indirect_dispatch.cpp)
      target_include_directories(test_live_compute_indirect_dispatch PRIVATE tests frontends)
      target_link_libraries(test_live_compute_indirect_dispatch PRIVATE prosper_live_renderer)
      add_test(NAME live_compute_indirect_dispatch COMMAND test_live_compute_indirect_dispatch)
      set_tests_properties(live_compute_indirect_dispatch PROPERTIES TIMEOUT 120)

      add_test(NAME game_compute_exec COMMAND test_game_compute)
      add_test(NAME game_compute_rtt_destination_mirror COMMAND test_game_compute_rtt_mirror)
      add_test(NAME game_compute_rtt_destination_mirror_no_direct_retile
               COMMAND test_game_compute_rtt_mirror)
      set_tests_properties(game_compute_rtt_destination_mirror_no_direct_retile PROPERTIES
        ENVIRONMENT "PROSPER_NO_DIRECT_IMAGE_RETILE=1")
      add_test(NAME game_compute_rtt_destination_mirror_no_rgba16
               COMMAND test_game_compute_rtt_mirror)
      set_tests_properties(game_compute_rtt_destination_mirror_no_rgba16 PROPERTIES
        ENVIRONMENT "PROSPER_NO_RGBA16_COMPUTE_RTT_MIRROR=1")
      add_test(NAME game_compute_rtt_destination_mirror_no_read_alias
               COMMAND test_game_compute_rtt_mirror)
      set_tests_properties(game_compute_rtt_destination_mirror_no_read_alias PROPERTIES
        ENVIRONMENT "PROSPER_NO_COMPUTE_RTT_READ_ALIAS=1")
      add_test(NAME game_compute_rtt_destination_mirror_no_folded_read_alias
               COMMAND test_game_compute_rtt_mirror)
      set_tests_properties(game_compute_rtt_destination_mirror_no_folded_read_alias PROPERTIES
        ENVIRONMENT "PROSPER_NO_FOLDED_COMPUTE_RTT_READ_ALIAS=1")
      add_test(NAME game_compute_rtt_destination_mirror_no_one_layer_array
               COMMAND test_game_compute_rtt_mirror)
      set_tests_properties(game_compute_rtt_destination_mirror_no_one_layer_array PROPERTIES
        ENVIRONMENT "PROSPER_NO_COMPUTE_RTT_ONE_LAYER_ARRAY_DEST=1")
      add_test(NAME game_compute_exec_no_adaptive_storage_result COMMAND test_game_compute)
      set_tests_properties(game_compute_exec_no_adaptive_storage_result PROPERTIES
        ENVIRONMENT "PROSPER_NO_ADAPTIVE_STORAGE_RESULT_VALIDATION=1")
      # PROSPER_NO_SKIP_SEED forces a real seed upload; the fixture asserts that mode's own expectations
      # (identical output is rewritten) instead of the default policy's (#3390).
      add_test(NAME game_compute_exec_no_skip_seed COMMAND test_game_compute)
      set_tests_properties(game_compute_exec_no_skip_seed PROPERTIES
        ENVIRONMENT "PROSPER_NO_SKIP_SEED=1")
      # The compute path's full-surface seed/pack intermediates are pooled rather than freshly
      # allocated. Their old allocation was `make_unique_for_overwrite`, i.e. nominally
      # indeterminate -- but at these sizes it was a fresh mmap, which reads as ZERO, so a call site
      # could be depending on zeroing it never asked for. This arm hands every lease 0xCD instead:
      # it fails exactly when such a dependency exists, which is the only way that claim is testable
      # against the real backend rather than argued from the diff.
      add_test(NAME game_compute_exec_scratch_poison COMMAND test_game_compute)
      set_tests_properties(game_compute_exec_scratch_poison PROPERTIES
        ENVIRONMENT "PROSPER_DECODE_SCRATCH_POISON=1")

      # #3464: registered on Windows as well as in the UNIX block below, deliberately. The tier it
      # exercises exists for hosts that cannot supply a 64-lane fragment subgroup -- NVIDIA is the
      # one this project runs on -- so a UNIX-only registration would execute it on RADV, where the
      # tier is never reached, and on lavapipe, but never on the hardware it was written for.
      add_executable(test_fragment_partial_wave_exec tests/gpu/execute/test_fragment_partial_wave_exec.cpp)
      target_include_directories(test_fragment_partial_wave_exec PRIVATE tests frontends)
      target_link_libraries(test_fragment_partial_wave_exec PRIVATE prosper_live_renderer)
      add_test(NAME fragment_partial_wave_exec COMMAND test_fragment_partial_wave_exec)
      set_tests_properties(fragment_partial_wave_exec PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 60)

      add_executable(test_fragment_numeric_mbcnt tests/gpu/execute/test_fragment_numeric_mbcnt.cpp)
      target_include_directories(test_fragment_numeric_mbcnt PRIVATE tests frontends)
      target_link_libraries(test_fragment_numeric_mbcnt PRIVATE prosper_live_renderer)
      add_test(NAME fragment_numeric_mbcnt COMMAND test_fragment_numeric_mbcnt)
      set_tests_properties(fragment_numeric_mbcnt PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 60)

      add_executable(test_fragment_proven_votes tests/gpu/execute/test_fragment_proven_votes.cpp)
      target_include_directories(test_fragment_proven_votes PRIVATE tests frontends)
      target_link_libraries(test_fragment_proven_votes PRIVATE prosper_live_renderer)
      add_executable(test_fragment_vote_refusal_reporting tests/gpu/execute/test_fragment_vote_refusal_reporting.cpp)
      target_include_directories(test_fragment_vote_refusal_reporting PRIVATE tests frontends)
      target_link_libraries(test_fragment_vote_refusal_reporting PRIVATE prosper_live_renderer)
      add_test(NAME fragment_vote_refusal_reporting COMMAND test_fragment_vote_refusal_reporting)
      set_tests_properties(fragment_vote_refusal_reporting PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 60)
      add_test(NAME fragment_proven_votes COMMAND test_fragment_proven_votes)
      set_tests_properties(fragment_proven_votes PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 60)
      add_test(NAME fragment_proven_votes_no_robust2 COMMAND test_fragment_proven_votes --expect-no-robust2)
      set_tests_properties(fragment_proven_votes_no_robust2 PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 60
        ENVIRONMENT "PROSPER_NO_ROBUST_BUFFER_ACCESS2=1")

      add_executable(test_numeric_mbcnt tests/gpu/execute/test_numeric_mbcnt.cpp)

      target_include_directories(test_numeric_mbcnt PRIVATE tests)
      target_link_libraries(test_numeric_mbcnt PRIVATE prosper_core Vulkan::Vulkan)
      add_test(NAME numeric_mbcnt COMMAND test_numeric_mbcnt)
      set_tests_properties(numeric_mbcnt PROPERTIES TIMEOUT 60)

      add_executable(test_numeric_wqm tests/gpu/execute/test_numeric_wqm.cpp)
      target_include_directories(test_numeric_wqm PRIVATE tests)
      target_link_libraries(test_numeric_wqm PRIVATE prosper_core Vulkan::Vulkan)
      add_test(NAME numeric_wqm COMMAND test_numeric_wqm)
      set_tests_properties(numeric_wqm PROPERTIES TIMEOUT 90)

      add_executable(test_portable_wave64_mask_reductions
                     tests/gpu/execute/test_portable_wave64_mask_reductions.cpp)
      target_include_directories(test_portable_wave64_mask_reductions PRIVATE tests)
      target_link_libraries(test_portable_wave64_mask_reductions PRIVATE prosper_core Vulkan::Vulkan)
      add_test(NAME portable_wave64_mask_reductions COMMAND test_portable_wave64_mask_reductions)
      set_tests_properties(portable_wave64_mask_reductions PROPERTIES TIMEOUT 60)

      add_executable(test_portable_wave64_readfirstlane
                     tests/gpu/execute/test_portable_wave64_readfirstlane.cpp)
      target_include_directories(test_portable_wave64_readfirstlane PRIVATE tests)
      target_link_libraries(test_portable_wave64_readfirstlane PRIVATE prosper_core Vulkan::Vulkan)
      add_test(NAME portable_wave64_readfirstlane COMMAND test_portable_wave64_readfirstlane)
      set_tests_properties(portable_wave64_readfirstlane PROPERTIES TIMEOUT 60)

      add_executable(test_portable_bpermute_execution tests/gpu/execute/test_portable_bpermute.cpp)
      target_include_directories(test_portable_bpermute_execution PRIVATE tests)
      target_link_libraries(test_portable_bpermute_execution PRIVATE prosper_core Vulkan::Vulkan)
      add_test(NAME portable_bpermute_execution COMMAND test_portable_bpermute_execution)
      set_tests_properties(portable_bpermute_execution PROPERTIES TIMEOUT 90)

      # #3425: registered on Windows as well as in the UNIX block below, deliberately. This
      # defect is about the WINDOWS/NVIDIA frontend's _Exit path and the cost it recovers was
      # measured there (#3450), so a Linux-only guard would test the fix everywhere except
      # where it was found. Both registrations run the same fixture and driver script.
      if(Python3_Interpreter_FOUND)
        add_executable(test_compute_cache_persistence tests/gpu/execute/test_compute_cache_persistence.cpp)
        target_include_directories(test_compute_cache_persistence PRIVATE tests frontends)
        target_link_libraries(test_compute_cache_persistence PRIVATE prosper_live_renderer)
        add_test(NAME compute_cache_persistence
                 COMMAND "${Python3_EXECUTABLE}"
                         "${CMAKE_CURRENT_SOURCE_DIR}/tests/gpu/execute/test_compute_cache_persistence.py"
                         $<TARGET_FILE:test_compute_cache_persistence>
                         "${CMAKE_CURRENT_SOURCE_DIR}/frontends/prosper-app/main.cpp")
        set_tests_properties(compute_cache_persistence PROPERTIES TIMEOUT 300)
      endif()

      # Local-only realized-submit replayer (#514), on Windows since #2331. Same target as the UNIX
      # block below, including the -Werror=switch gate: gpu_replay maps LiveTargetPixelFormat itself
      # in inspect_live_target, and an unhandled member there falls through to the seed's Rgba8Unorm
      # default and makes --inspect-only mislabel the seed -- sending an investigation down the wrong
      # path using its own instrument. That failure mode is not platform-specific, so neither is the
      # gate.
      add_executable(gpu_replay tools/gpu_replay/gpu_replay.cpp)
      # Spells a `shared/...` include, which resolves from `frontends`.
      target_include_directories(gpu_replay PRIVATE frontends)
      target_include_directories(gpu_replay PRIVATE src tests)
      target_link_libraries(gpu_replay PRIVATE prosper_core prosper_live_renderer Vulkan::Vulkan)
      if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        # -Werror=format for the same reason as -Werror=switch above, and found the same way: by a
        # defect this tool's own diagnostics hid. gpu_replay prints index values with %zu, and #2836
        # gave three of those printf arguments a distinct type (ItemIndex) while the format stayed
        # %zu. Varargs performs no conversion, so the type system has nothing to check, and this
        # target compiled with neither -Wall nor -Wformat -- the mismatch was invisible in the build
        # and was caught only in review. #2840 will type the operation indices too, and gpu_replay.cpp
        # prints those with %zu at roughly sixteen more sites, so the class recurs the moment that
        # lands. Measured clean under -Wformat=2 before this gate was added.
        #
        # THIS GATE BINDS ON GCC ONLY, despite the GNU|Clang guard above. Clang deliberately looks
        # through a scoped enum to its underlying type here and stays silent at -Wformat,
        # -Werror=format, -Wformat=2 AND -Wall -Wextra; only -Wformat-pedantic makes it fire. The
        # same positive control that errors under g++ exits 0 under clang++. It still protects every
        # build this project actually performs -- both the Linux and the MinGW Windows legs are GCC,
        # and the control was verified to fire on both -- but do not read the guard as Clang cover.
        # -Wformat-pedantic on a Clang-only branch was measured clean on this TU if that is ever
        # wanted; it is left out because no build here uses Clang and an unrecognized option is its
        # own hazard. Deliberately NOT widened to -Wformat=2: GCC already diagnoses this class at
        # plain -Wformat, so =2 adds only unrelated surface, and it does not even mean the same thing
        # to the two compilers (GCC gives -Wformat-nonliteral where Clang gives
        # -Wmissing-format-attribute, which GCC lacks entirely).
        target_compile_options(gpu_replay PRIVATE -Werror=switch -Werror=format)
      endif()
    else()
      message(STATUS "Vulkan not found — Windows boot_trace built without the live renderer")
    endif()
  endif()

  # M2+: real memory mapping, trap, boot, and the boot-trace debug tool (Linux + macOS/Rosetta).
  if(UNIX)
    add_executable(boot_trace tools/boot_trace/boot_trace.cpp tools/boot_trace/first_frame_capture.cpp)
    # Includes a shared fixture, which resolves from the `tests` root.
    target_include_directories(boot_trace PRIVATE tests)
    # Spells a `shared/...` include, which resolves from `frontends`.
    target_include_directories(boot_trace PRIVATE frontends)
    target_link_libraries(boot_trace PRIVATE prosper_core)

    if(NOT APPLE)
      # Zero-dependency Linux evdev gamepad frontend (outside prosper_core). Always available on Linux;
      # boot_trace installs it (gated by PROSPER_PAD) when the SDL3 gamepad frontend isn't built.
      add_library(prosper_pad_evdev STATIC frontends/pad_evdev/pad_evdev.cpp)
      target_include_directories(prosper_pad_evdev PUBLIC frontends/pad_evdev src)
      target_link_libraries(prosper_pad_evdev PUBLIC prosper_core)
      target_link_libraries(boot_trace PRIVATE prosper_pad_evdev)
      target_compile_definitions(boot_trace PRIVATE PROSPER_PAD_EVDEV=1)
    endif()

    add_executable(imgdump tools/imgdump/imgdump.cpp)
    target_link_libraries(imgdump PRIVATE prosper_core)

    add_executable(test_trap_linux tests/misc/test_trap_linux.cpp)
    target_link_libraries(test_trap_linux PRIVATE prosper_core)
    if(HAVE_GAME_DUMP)
      add_test(NAME trap_identifies_imports
               COMMAND test_trap_linux "${GAME_DUMP}")
    endif()

    add_executable(test_boot_linux tests/misc/test_boot_linux.cpp)
    target_link_libraries(test_boot_linux PRIVATE prosper_core)
    if(HAVE_GAME_DUMP)
      # Messenger-specific: links THIS title's own Il2cppUserAssemblies.prx / PS5Util.prx (#1573).
      prosper_add_title_gated_test(boot_reaches_first_syscall PPSA24651
               test_boot_linux "${GAME_DUMP}")
    endif()

    if(NOT APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64)$")
      # Real runtime PRX loading (#639). Hermetic: the test writes its own synthetic PRX outside
      # Media/Plugins, so it needs no game dump — but it maps that module at the fixed runtime-module
      # base and EXECUTES its module_start, whose bytes the test emits by hand, so it is host-x86-64
      # only. Its scratch root goes in the build tree, never /tmp — no path is passed here, so the
      # test derives one per process from tests/fixtures/test_scratch.h. Passing a fixed
      # "${CMAKE_CURRENT_BINARY_DIR}/runtime-prx-test-root" is what #2582 was about: two ctest runs
      # against one build directory then shared the .prx they each write and map.
      add_executable(test_runtime_prx_load tests/host/image/test_runtime_prx_load.cpp)
      # Includes a shared fixture, which resolves from the `tests` root.
      target_include_directories(test_runtime_prx_load PRIVATE tests)
      target_link_libraries(test_runtime_prx_load PRIVATE prosper_core)
      add_test(NAME runtime_prx_load COMMAND test_runtime_prx_load)
      if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        # Initial installation must reject an oversized last-slot hook before touching live pages.
        prosper_add_gtest(test_initial_stub_install
          SOURCES tests/host/image/test_initial_stub_install.cpp
          LIBRARIES prosper_core
          PROPERTIES TIMEOUT 20)
        target_link_options(test_initial_stub_install PRIVATE "-Wl,--wrap=mmap")
        target_link_options(test_runtime_prx_load PRIVATE "-Wl,--wrap=mmap")
        foreach(failure map image emission)
          add_test(NAME runtime_prx_append_${failure}_failure
                   COMMAND test_runtime_prx_load --append-${failure}-failure)
          set_tests_properties(runtime_prx_append_${failure}_failure PROPERTIES TIMEOUT 20)
        endforeach()
        add_test(NAME runtime_prx_missing_data_base COMMAND test_runtime_prx_load --missing-data-base)
        set_tests_properties(runtime_prx_missing_data_base PROPERTIES TIMEOUT 20)
      endif()
    endif()

    # Regression test for the real setjmp/longjmp impl (x86-64 asm; used by the GC's root scan).
    prosper_add_gtest(test_setjmp
      SOURCES tests/test_setjmp.cpp
      LIBRARIES prosper_core)

    prosper_add_gtest(test_stack_registry
      SOURCES tests/host/image/test_stack_registry.cpp
      LIBRARIES prosper_core)

    prosper_add_gtest(test_rwlock_once
      SOURCES tests/hle/test_rwlock_once.cpp
      LIBRARIES prosper_core)

    # #2042: a destroyed guest sync object's storage is quarantined, not returned to the allocator,
    # so a Destroy racing a handler cannot corrupt prosper's own heap. The assertions hold on every
    # allocator, so the test itself runs everywhere.
    add_executable(test_sync_destroy_lifetime tests/hle/sync/test_sync_destroy_lifetime.cpp)
    target_link_libraries(test_sync_destroy_lifetime PRIVATE prosper_core)
    add_test(NAME sync_destroy_lifetime COMMAND test_sync_destroy_lifetime)
    set_tests_properties(sync_destroy_lifetime PROPERTIES TIMEOUT 120)

    # The window switch must FAIL SAFE. `atof` returns 0.0 on no conversion, so an early version of
    # retire_window_seconds() turned the guard off for `=on`, `=true` or any typo — the exact
    # use-after-free this exists to prevent, reached by misspelling a tuning knob. This arm passes an
    # unparseable value and requires the suite to still PASS, i.e. the default window is in force.
    # It asserts the green outcome, so unlike the counter-arm it is allocator-independent.
    add_test(NAME sync_destroy_lifetime_bad_window COMMAND test_sync_destroy_lifetime)
    set_tests_properties(sync_destroy_lifetime_bad_window PROPERTIES
        ENVIRONMENT "PROSPER_SYNC_RETIRE_SECONDS=on" TIMEOUT 120)

    # The counter-arm runs the SAME binary with the quarantine switched off
    # (PROSPER_SYNC_RETIRE_SECONDS=0 is the pre-#2042 immediate free) and requires it to FAIL, so CI
    # keeps proving the assertions can move without anyone rebuilding against an older master.
    #
    # LINUX ONLY, and the restriction is the honest one rather than a convenience: the arm is red
    # because glibc's tcache hands a just-freed block of the same size back to the very next
    # allocation, deterministically and LIFO. That reuse is the mechanism being demonstrated, not an
    # incidental detail — on an allocator that does not reuse promptly the assertions would pass and
    # WILL_FAIL would turn a correct result into a red build. Asserting an allocator's internals as
    # a merge gate on hosts where they were never measured is how a guard starts failing for reasons
    # unrelated to what it guards.
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
      add_test(NAME sync_destroy_lifetime_counter_arm COMMAND test_sync_destroy_lifetime)
      set_tests_properties(sync_destroy_lifetime_counter_arm PROPERTIES
          ENVIRONMENT "PROSPER_SYNC_RETIRE_SECONDS=0" WILL_FAIL TRUE TIMEOUT 120)
    endif()
    # #2178: every fallible scePthread* spelling reports the libkernel-encoded error while the POSIX
    # spelling on the same body keeps the bare FreeBSD errno. Asserts both halves per entry point —
    # a single-spelling arm cannot see a missing (or a wrongly added) SCE_PTHREAD_ALIAS.
    prosper_add_gtest(test_pthread_error_encoding
      SOURCES tests/hle/kernel/test_pthread_error_encoding.cpp
      LIBRARIES prosper_core)

    # #2596: the C11 _Mtx_* / _Cnd_* handlers must resolve a guest sync slot the way the libkernel
    # entry point they forward to on hardware resolves it. Asserts what the handler DID (blocked,
    # excluded a second thread, self-initialised the slot), never what it returned -- _Cnd_wait's
    # unconditional 0 is the shipped contract. TIMEOUT is generous because every arm is a bounded
    # spin: a regression here is a thread that never wakes, and each arm reports it as a failure.
    add_executable(test_c11_thread_slots tests/hle/test_c11_thread_slots.cpp)
    target_link_libraries(test_c11_thread_slots PRIVATE prosper_core)
    add_test(NAME c11_thread_slots COMMAND test_c11_thread_slots)
    set_tests_properties(c11_thread_slots PROPERTIES TIMEOUT 120)

    # __tls_get_addr per-thread DTV is purged (blocks freed) on both thread-exit paths (#68), and
    # normal return leaves guest %fs before glibc resumes host pthread cleanup (#644).
    prosper_add_gtest(test_tls_dtv_purge
      SOURCES tests/hle/test_tls_dtv_purge.cpp
      LIBRARIES prosper_core)

    # #2613: nothing joins guest threads at shutdown, so thread_trampoline's tail runs while the
    # process is already inside __run_exit_handlers and touches the thread-name / thread-stack / DTV
    # registries. They must be process-lifetime (src/host/platform/immortal.hpp), i.e. register no static
    # destructor. The test parks a guest worker and drives the whole tail from an exit-time probe, so
    # the post-destruction access happens on EVERY run instead of 1-in-300 — and the probe checks the
    # two library canaries first, so a broken ordering assumption fails instead of passing vacuously.
    add_executable(test_exit_registry_lifetime tests/host/image/test_exit_registry_lifetime.cpp)
    target_link_libraries(test_exit_registry_lifetime PRIVATE prosper_core)
    add_test(NAME exit_registry_lifetime COMMAND test_exit_registry_lifetime)
    set_tests_properties(exit_registry_lifetime PROPERTIES TIMEOUT 120)

    prosper_add_gtest(test_agc_dcb
      SOURCES tests/gpu/pm4/test_agc_dcb.cpp
      LIBRARIES prosper_core)

    # #1143: each sceAgc*GetSize must equal its builder's emitted dword count — drift guard.
    prosper_add_gtest(test_agc_getsize
      SOURCES tests/hle/test_agc_getsize.cpp
      LIBRARIES prosper_core)

    # #1748: a builder must not emit MORE dwords than the hardware packet it stands for — the guest
    # reserves its command buffer from the real AGC sizes, so an oversized packet overruns it.
    prosper_add_gtest(test_agc_epilogue_window
      SOURCES tests/gpu/pm4/test_agc_epilogue_window.cpp
      LIBRARIES prosper_core)

    # libSceVideoOut flip->present: SubmitFlip scans out the registered buffer; present_readback
    # returns the presented frame's real pixels. The surface the renderer presents frames to.
    prosper_add_gtest(test_present
      SOURCES tests/misc/test_present.cpp
      LIBRARIES prosper_core)

    # Graphics-phase foundation + agentic-first verification harness: headless offscreen Vulkan
    # (render/clear -> readback -> pixel/CRC assert). Only built if Vulkan dev files are present.
    if(NOT TARGET Vulkan::Vulkan)
      find_package(Vulkan QUIET)
    endif()
    if(Vulkan_FOUND)
      # Live game rendering: with Vulkan available, boot_trace can register the live submit renderer
      # (PROSPER_RENDER=1) so the game's SubmitDcb draws execute + present + dump frames to disk.
      target_link_libraries(boot_trace PRIVATE Vulkan::Vulkan)
      target_compile_definitions(boot_trace PRIVATE PROSPER_HAVE_VULKAN=1)

      # Shared live renderer (frontends/shared): the DrawItem->Vulkan compositor, factored out of
      # boot_trace so BOTH boot_trace and prosper-app register the SAME renderer instead of
      # duplicating it. Needs render_runner.h (tests/) and refvs.inc (tools/boot_trace/).
      add_library(prosper_live_renderer STATIC frontends/shared/live/live_renderer.cpp
                                               frontends/shared/live/submit_renderer/callback_state.cpp
                                               frontends/shared/live/submit_renderer/callback_thread_state.cpp
                                               frontends/shared/live/submit_renderer/draw_resources.cpp
                                               frontends/shared/live/submit_renderer/image_resources.cpp
                                               frontends/shared/live/submit_renderer/timing_report.cpp
                                               frontends/shared/live/submit_renderer/backend_draws.cpp
                                               frontends/shared/live/submit_renderer/backend_timing.cpp
                                               frontends/shared/live/submit_renderer/per_target_passes.cpp
                                               frontends/shared/live/submit_renderer/resolve_pass.cpp
                                               frontends/shared/live/submit_renderer/pass_diagnostics.cpp
                                               frontends/shared/live/submit_renderer/callback_prelude.cpp
                                               frontends/shared/live/submit_renderer/callback_diagnostics.cpp
                                               frontends/shared/live/submit_renderer/single_framebuffer.cpp
                                               frontends/shared/live/submit_renderer/final_span_present.cpp
                                               frontends/shared/live/live_compute.cpp
                                               frontends/shared/present/present_blit.cpp
                                               frontends/shared/present/compute_scanout.cpp)
      target_include_directories(prosper_live_renderer PRIVATE src tests tools/boot_trace)
      target_link_libraries(prosper_live_renderer PUBLIC prosper_core Vulkan::Vulkan
                                                   prosper_performance_capture)
      target_include_directories(prosper_live_renderer PUBLIC frontends)
      # A LiveTargetPixelFormat mapping that quietly falls back to RGBA8 has now cost two titles a
      # whole render layer (#773 Dead Cells, and Syberia's black 3D menu). frontends/shared/
      # live_target_format.hpp maps the enum with default-less switches; -Werror=switch promotes a
      # member added without updating them from a silent wrong format to a build failure.
      if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(prosper_live_renderer PRIVATE -Werror=switch)
      endif()
      target_link_libraries(boot_trace PRIVATE prosper_live_renderer)

      # vkprobe (#2945/#2937): a CONTROL that AVOIDS prosper's host-side Vulkan usage rather than
      # reproducing it, so a failure here shows that usage is not NECESSARY for the defect -- never
      # that it is exonerated. Drives a dumped SPIR-V pair through a bare Vulkan pipeline with no
      # prosper code in the process. It does not clear the recompiler (the SPIR-V is prosper's own
      # output) and it does not clear the driver -- run enough times it reproduces #2945 itself.
      # Read tools/vkprobe/README.md before quoting a clean run. Deliberately links nothing from
      # prosper.
      add_executable(vkprobe tools/vkprobe/vkprobe.cpp)
      target_link_libraries(vkprobe PRIVATE Vulkan::Vulkan)

      # Local-only realized-submit replayer (#514): no guest boot, reads a versioned capture whose game
      # data remains gitignored, renders it through the same live backend, and checks the captured hash.
      add_executable(gpu_replay tools/gpu_replay/gpu_replay.cpp)
      # Spells a `shared/...` include, which resolves from `frontends`.
      target_include_directories(gpu_replay PRIVATE frontends)
      target_include_directories(gpu_replay PRIVATE src tests)
      target_link_libraries(gpu_replay PRIVATE prosper_core prosper_live_renderer Vulkan::Vulkan)
      # gpu_replay maps LiveTargetPixelFormat itself (inspect_live_target), and it is the tool an
      # investigation reaches for FIRST. An unhandled member there falls through to the
      # GpuCaptureRttSeed::format default (Rgba8Unorm) and makes --inspect-only mislabel the seed,
      # sending the next investigation down the wrong path using its own instrument. Same gate as
      # prosper_live_renderer so the exhaustiveness contract holds at every mapping, not just the
      # ones inside frontends/shared/live/live_target_format.hpp.
      if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        # -Werror=format for the same reason as -Werror=switch above, and found the same way: by a
        # defect this tool's own diagnostics hid. gpu_replay prints index values with %zu, and #2836
        # gave three of those printf arguments a distinct type (ItemIndex) while the format stayed
        # %zu. Varargs performs no conversion, so the type system has nothing to check, and this
        # target compiled with neither -Wall nor -Wformat -- the mismatch was invisible in the build
        # and was caught only in review. #2840 will type the operation indices too, and gpu_replay.cpp
        # prints those with %zu at roughly sixteen more sites, so the class recurs the moment that
        # lands. Measured clean under -Wformat=2 before this gate was added.
        #
        # THIS GATE BINDS ON GCC ONLY, despite the GNU|Clang guard above. Clang deliberately looks
        # through a scoped enum to its underlying type here and stays silent at -Wformat,
        # -Werror=format, -Wformat=2 AND -Wall -Wextra; only -Wformat-pedantic makes it fire. The
        # same positive control that errors under g++ exits 0 under clang++. It still protects every
        # build this project actually performs -- both the Linux and the MinGW Windows legs are GCC,
        # and the control was verified to fire on both -- but do not read the guard as Clang cover.
        # -Wformat-pedantic on a Clang-only branch was measured clean on this TU if that is ever
        # wanted; it is left out because no build here uses Clang and an unrecognized option is its
        # own hazard. Deliberately NOT widened to -Wformat=2: GCC already diagnoses this class at
        # plain -Wformat, so =2 adds only unrelated surface, and it does not even mean the same thing
        # to the two compilers (GCC gives -Wformat-nonliteral where Clang gives
        # -Wmissing-format-attribute, which GCC lacks entirely).
        target_compile_options(gpu_replay PRIVATE -Werror=switch -Werror=format)
      endif()

      # Seed flags are renderer-registration inputs. Exercise the real gpu_replay process so an
      # early --recompile-raw registration cannot pass by proving substitution but failing restore.
      add_executable(test_gpu_replay_seed_order tests/gpu/capture/test_gpu_replay_seed_order.cpp)
      target_include_directories(test_gpu_replay_seed_order PRIVATE src tests)
      target_link_libraries(test_gpu_replay_seed_order PRIVATE prosper_core)
      add_test(NAME gpu_replay_seeded_raw_execution
        COMMAND test_gpu_replay_seed_order $<TARGET_FILE:gpu_replay>)

      # Exercise the real metadata-only CLI path for v46's selected raw resource-input witness.
      # Unit tests cover verdict policy; this process test keeps inspect formatting from silently
      # disappearing or collapsing full-match/mismatch/unavailable into one output state.
      add_test(NAME gpu_replay_resource_input_inspect
        COMMAND test_gpu_capture --inspect-resource-input $<TARGET_FILE:gpu_replay>)

      # Capture -> serialize -> owned replay backing -> SAME live renderer -> pixels. This specifically
      # proves replay textures do not dereference the captured guest VA and still sample exact bytes.
      add_executable(test_gpu_capture_render tests/misc/test_gpu_capture_render.cpp)
      add_executable(test_depth_cube_source_snapshot tests/shared/live/test_depth_cube_source_snapshot.cpp)
      target_include_directories(test_depth_cube_source_snapshot PRIVATE frontends src tests)
      target_link_libraries(test_depth_cube_source_snapshot PRIVATE prosper_core prosper_live_renderer Vulkan::Vulkan)
      add_test(NAME depth_cube_source_snapshot COMMAND test_depth_cube_source_snapshot)
      add_test(NAME depth_cube_source_snapshot_control COMMAND test_depth_cube_source_snapshot --control)
      add_test(NAME depth_cube_source_snapshot_scalar COMMAND test_depth_cube_source_snapshot)
      # These arms assert the CPU cube bridge's cache and snapshot accounting, so they pin the
      # CPU route; the GPU route for fully retained cubes has its own test below.
      set_tests_properties(depth_cube_source_snapshot PROPERTIES
        ENVIRONMENT "PROSPER_GPU_REPLAY_DS_SEEDS=1;PROSPER_NO_GPU_DEPTH_CUBE=1")
      set_tests_properties(depth_cube_source_snapshot_control PROPERTIES
        ENVIRONMENT "PROSPER_GPU_REPLAY_DS_SEEDS=1;PROSPER_NO_CUBE_SOURCE_SNAPSHOT_PRUNING=1;PROSPER_NO_GPU_DEPTH_CUBE=1")
      set_tests_properties(depth_cube_source_snapshot_scalar PROPERTIES
        ENVIRONMENT "PROSPER_GPU_REPLAY_DS_SEEDS=1;PROSPER_NO_DEPTH_CUBE_SIMD=1;PROSPER_NO_GPU_DEPTH_CUBE=1")
      # GPU-resident retained depth cube (#3873): bit-exact against the CPU bridge, the conversion's
      # out-of-range domain, and live-renderer routing; --control pins PROSPER_NO_GPU_DEPTH_CUBE=1.
      add_executable(test_depth_cube_gpu tests/shared/live/test_depth_cube_gpu.cpp)
      target_include_directories(test_depth_cube_gpu PRIVATE frontends src tests)
      target_link_libraries(test_depth_cube_gpu PRIVATE prosper_core prosper_live_renderer Vulkan::Vulkan)
      add_test(NAME depth_cube_gpu COMMAND test_depth_cube_gpu)
      add_test(NAME depth_cube_gpu_control COMMAND test_depth_cube_gpu --control)
      add_executable(test_depth_array_source_snapshot tests/shared/live/test_depth_array_source_snapshot.cpp)
      target_include_directories(test_depth_array_source_snapshot PRIVATE frontends src tests)
      target_link_libraries(test_depth_array_source_snapshot PRIVATE prosper_core prosper_live_renderer Vulkan::Vulkan)
      add_test(NAME depth_array_source_snapshot COMMAND test_depth_array_source_snapshot)
      add_test(NAME depth_array_source_snapshot_per_draw_control
        COMMAND test_depth_array_source_snapshot --per-draw-control)
      add_test(NAME depth_array_source_snapshot_expanded_control
        COMMAND test_depth_array_source_snapshot --expanded-control)
      add_test(NAME depth_array_source_snapshot_expanded_per_draw_control
        COMMAND test_depth_array_source_snapshot --expanded-control --per-draw-control)
      # The same checks through the GPU-resident retained-array copy (#3873): identical sampled
      # pixels, zero CPU upload, and GPU-copied bytes matching the compact/expanded representation.
      add_test(NAME depth_array_source_snapshot_gpu COMMAND test_depth_array_source_snapshot --gpu)
      add_test(NAME depth_array_source_snapshot_gpu_per_draw_control
        COMMAND test_depth_array_source_snapshot --gpu --per-draw-control)
      add_test(NAME depth_array_source_snapshot_gpu_expanded_control
        COMMAND test_depth_array_source_snapshot --gpu --expanded-control)
      add_test(NAME depth_array_source_snapshot_gpu_expanded_per_draw_control
        COMMAND test_depth_array_source_snapshot --gpu --expanded-control --per-draw-control)
      add_test(NAME backend_texture_path_census_budget
        COMMAND test_depth_array_source_snapshot --texture-path-census-budget)
      # Windows environment APIs erase a variable assigned the empty string, so this distinct
      # empty-vs-unset contract has an executable arm only where the process can observe it.
      if(NOT WIN32)
        add_test(NAME backend_texture_path_census_empty COMMAND ${CMAKE_COMMAND} -E env
          PROSPER_BACKEND_TEXTURE_PATH_CENSUS=1
          PROSPER_BACKEND_TEXTURE_PATH_CENSUS_MIN_DRAWS=
          $<TARGET_FILE:test_depth_array_source_snapshot> --texture-path-census-empty)
      endif()
      set_tests_properties(backend_texture_path_census_budget
        PROPERTIES ENVIRONMENT "PROSPER_BACKEND_TEXTURE_PATH_CENSUS=1")
      set_tests_properties(backend_texture_path_census_budget PROPERTIES ENVIRONMENT
        "PROSPER_BACKEND_TEXTURE_PATH_CENSUS=1;PROSPER_BACKEND_TEXTURE_PATH_CENSUS_MIN_DRAWS=10")
      add_executable(test_depth_only_attachment_extent tests/shared/live/test_depth_only_attachment_extent.cpp)
      target_include_directories(test_depth_only_attachment_extent PRIVATE frontends src tests)
      target_link_libraries(test_depth_only_attachment_extent PRIVATE prosper_core prosper_live_renderer Vulkan::Vulkan)
      add_test(NAME depth_only_attachment_extent COMMAND test_depth_only_attachment_extent)
      # #3907: the same binary on the CPU readback path (live colour targets off).
      add_test(NAME depth_only_attachment_extent_readback
               COMMAND test_depth_only_attachment_extent --readback-path)
      set_tests_properties(depth_only_attachment_extent_readback PROPERTIES
                           ENVIRONMENT "PROSPER_NO_LIVE_PERSISTENT_COLOR_TARGETS=1"
                           ENVIRONMENT_MODIFICATION
                           "PROSPER_NO_PERF_ALARMS=unset:;PROSPER_RTT_NOSEED=unset:;PROSPER_RTT_NOSEED_TARGET=unset:")
      # #3892: when the live renderer callback's prelude statics initialise, across two
      # registrations in one process. Own process: every arm needs untouched statics.
      add_executable(test_live_renderer_registration_statics
        tests/shared/live/test_live_renderer_registration_statics.cpp)
      target_include_directories(test_live_renderer_registration_statics PRIVATE frontends src tests)
      target_link_libraries(test_live_renderer_registration_statics PRIVATE prosper_core prosper_live_renderer Vulkan::Vulkan)
      add_test(NAME live_renderer_registration_statics COMMAND test_live_renderer_registration_statics)
      set_tests_properties(live_renderer_registration_statics PROPERTIES ENVIRONMENT_MODIFICATION
        "PROSPER_RENDER_DELAY_MS=unset:;PROSPER_TEXTURE_DECODE_CACHE_MB=unset:;PROSPER_RENDER_FIRST=unset:;PROSPER_RENDER_LAST=unset:;PROSPER_TEXTURE_VALIDATION_CENSUS=set:1")
      # #3892: real builder rejection/skip/continuation and census tail, with seeded manifests.
      add_executable(test_draw_resource_status tests/shared/live/test_draw_resource_status.cpp)
      # Maintained nonshipping TU role; all local diagnostic checks/inventory remain enabled.
      set_property(TARGET test_draw_resource_status PROPERTY PROSPER_DIAG_MEMBER_FACT_DOMAIN TEST_LOCAL)
      target_include_directories(test_draw_resource_status PRIVATE frontends src tests)
      target_link_libraries(test_draw_resource_status PRIVATE prosper_core prosper_live_renderer Vulkan::Vulkan)
      add_test(NAME draw_resource_status COMMAND test_draw_resource_status)
      set_tests_properties(draw_resource_status PROPERTIES
        ENVIRONMENT "PROSPER_TEXREF_CENSUS=1" TIMEOUT 45)
      prosper_add_gtest(test_float32_array_upload
        SOURCES tests/shared/live/test_float32_array_upload.cpp
        INCLUDES frontends src tests
        LIBRARIES prosper_core prosper_live_renderer Vulkan::Vulkan)
      add_executable(test_graphics_raw_wide_upload tests/shared/live/test_graphics_raw_wide_upload.cpp)
      target_include_directories(test_graphics_raw_wide_upload PRIVATE frontends src tests)
      target_link_libraries(test_graphics_raw_wide_upload PRIVATE prosper_core prosper_live_renderer Vulkan::Vulkan)
      add_test(NAME graphics_raw_wide_upload COMMAND test_graphics_raw_wide_upload)
      add_test(NAME graphics_nested_wide_cpu COMMAND test_graphics_raw_wide_upload --nested-cpu-only)
      add_test(NAME graphics_nested_wide_upload COMMAND test_graphics_raw_wide_upload --nested-only)
      set_tests_properties(graphics_nested_wide_cpu graphics_nested_wide_upload PROPERTIES
        SKIP_RETURN_CODE 77 TIMEOUT 60)
      add_executable(test_decoder_source_snapshot tests/shared/live/test_decoder_source_snapshot.cpp)
      target_include_directories(test_decoder_source_snapshot PRIVATE frontends src tests)
      target_link_libraries(test_decoder_source_snapshot PRIVATE prosper_core prosper_live_renderer Vulkan::Vulkan)
      # This test's BC1 arm pins the CPU DECODER's source-snapshot contract, so it keeps the decoder
      # even on a device that samples BCn natively (#3873; native BC is test_native_bc_upload's).
      add_test(NAME decoder_source_snapshot COMMAND test_decoder_source_snapshot)
      set_tests_properties(decoder_source_snapshot PROPERTIES
        ENVIRONMENT "PROSPER_NO_NATIVE_BC=1")
      add_test(NAME decoder_source_snapshot_control COMMAND test_decoder_source_snapshot --control)
      set_tests_properties(decoder_source_snapshot_control PROPERTIES
        ENVIRONMENT "PROSPER_NO_DECODE_SOURCE_SNAPSHOT_REUSE=1;PROSPER_NO_NATIVE_BC=1")
      if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        add_test(NAME decoder_source_snapshot_watch COMMAND test_decoder_source_snapshot --watch)
        set_tests_properties(decoder_source_snapshot_watch PROPERTIES
          ENVIRONMENT "PROSPER_TEXTURE_WRITE_WATCH_MIN_KB=1;PROSPER_NO_NATIVE_BC=1")
      endif()
      # #3873: native BCn sampled upload against the CPU decoder, with the opt-out as control.
      add_executable(test_native_bc_upload tests/shared/live/test_native_bc_upload.cpp)
      target_include_directories(test_native_bc_upload PRIVATE frontends src tests)
      target_link_libraries(test_native_bc_upload PRIVATE prosper_core prosper_live_renderer Vulkan::Vulkan)
      add_test(NAME native_bc_upload COMMAND test_native_bc_upload)
      add_test(NAME native_bc_upload_decoder_control COMMAND test_native_bc_upload --decoder)
      set_tests_properties(native_bc_upload_decoder_control PROPERTIES
        ENVIRONMENT "PROSPER_NO_NATIVE_BC=1")
      add_executable(test_texture_preparation tests/shared/live/test_texture_preparation.cpp)
      target_include_directories(test_texture_preparation PRIVATE frontends src tests)
      target_link_libraries(test_texture_preparation PRIVATE prosper_core prosper_live_renderer Vulkan::Vulkan)
      add_test(NAME texture_preparation COMMAND test_texture_preparation)
      add_test(NAME texture_preparation_control COMMAND test_texture_preparation --control)
      set_tests_properties(texture_preparation_control PROPERTIES
        ENVIRONMENT "PROSPER_NO_TEXTURE_PREPARATION_PRUNING=1")
      # Spells a `shared/...` include, which resolves from `frontends`.
      target_include_directories(test_gpu_capture_render PRIVATE frontends)
      target_include_directories(test_gpu_capture_render PRIVATE src tests tools/boot_trace)
      target_link_libraries(test_gpu_capture_render PRIVATE prosper_core prosper_live_renderer Vulkan::Vulkan)
      add_test(NAME gpu_capture_render_replay COMMAND test_gpu_capture_render)
      add_test(NAME gpu_capture_render_snapshot_copy COMMAND test_gpu_capture_render --expect-source-snapshot-copy)
      set_tests_properties(gpu_capture_render_snapshot_copy PROPERTIES
        ENVIRONMENT "PROSPER_NO_TEXTURE_SOURCE_SNAPSHOT_MOVE=1")

      # PROSPER_SKIP_DRAW_PROGRAM through the REAL live renderer: an unnamed program renders, the
      # named one is withheld, and a mixed submit loses only the named draw. Needs the live renderer
      # (and therefore Vulkan) because the wiring, not the selector logic, is what is under test.
      add_executable(test_draw_program_skip_render tests/gpu/diagnostics/test_draw_program_skip_render.cpp)
      target_include_directories(test_draw_program_skip_render PRIVATE frontends)
      target_include_directories(test_draw_program_skip_render PRIVATE src tests tools/boot_trace)
      target_link_libraries(test_draw_program_skip_render PRIVATE prosper_core prosper_live_renderer Vulkan::Vulkan)
      add_test(NAME draw_program_skip_render COMMAND test_draw_program_skip_render)

      # #3204: the renderer-target format table, extracted from execute_item where it was an
      # `||` chain -- the shape that cost Dead Cells an HDR target (#773) and Syberia its 3D menu.
      # Now an exhaustive switch: the compiler pins exhaustiveness, this test pins the table.
      # Registered in BOTH the UNIX and WIN32 branches, like the compute-backend tests above:
      # live_target_format.hpp includes <vulkan/vulkan.h>, so it needs an if(Vulkan_FOUND) scope,
      # and this file has one per platform.
      prosper_add_gtest(test_live_target_format_match
        SOURCES tests/gpu/test_live_target_format_match.cpp
        INCLUDES frontends src
        LIBRARIES prosper_core Vulkan::Vulkan)
      # Armed from the ENVIRONMENT rather than by a setenv inside the test. The persistent-target
      # opt-out is read through PROSPER_ENV_VALUE, which caches, so arming it from inside the process
      # is exactly the vacuous-arm defect cached_env_arming_logic exists to catch; and the selector
      # itself is read once into a function-local static, so environment arming removes the ordering
      # argument for it too.
      set_tests_properties(draw_program_skip_render PROPERTIES ENVIRONMENT
        "PROSPER_SKIP_DRAW_PROGRAM=0x5006c6a00;PROSPER_NO_LIVE_PERSISTENT_COLOR_TARGETS=1")

      prosper_add_gtest(test_vulkan_offscreen
        SOURCES tests/test_vulkan_offscreen.cpp
        LIBRARIES Vulkan::Vulkan)

      # First real rendered output: a VkPipeline whose topology/format come from a RenderState via
      # vk_translate, drawing a triangle (embedded placeholder SPIR-V), verified by pixel readback.
      # Includes a shared fixture, which resolves from the `tests` root.
      add_executable(test_vulkan_triangle tests/gpu/state/test_vulkan_triangle.cpp)
      # Includes a shared fixture, which resolves from the `tests` root.
      target_include_directories(test_vulkan_triangle PRIVATE tests)
      target_link_libraries(test_vulkan_triangle PRIVATE prosper_core Vulkan::Vulkan)
      add_test(NAME vulkan_triangle_render COMMAND test_vulkan_triangle)

      # Execution-differential harness (verification layer 4): run a compute shader with known
      # inputs, assert numeric outputs. The RDNA2->SPIR-V recompiler's output will plug in here.
      # Includes a shared fixture, which resolves from the `tests` root.
      prosper_add_gtest(test_compute_exec
        SOURCES tests/test_compute_exec.cpp
        INCLUDES tests
        LIBRARIES Vulkan::Vulkan)

      # Execute a real game kernel with its user/system-register ABI and resource table.
      add_executable(test_game_compute tests/gpu/recompiler/test_game_compute.cpp)
      # Includes a shared fixture, which resolves from the `tests` root.
      target_include_directories(test_game_compute PRIVATE tests)
      # Spells a `shared/...` include, which resolves from `frontends`.
      target_include_directories(test_game_compute PRIVATE frontends)
      target_link_libraries(test_game_compute PRIVATE prosper_live_renderer)
      add_executable(test_game_compute_rtt_mirror tests/gpu/recompiler/test_game_compute_rtt_mirror.cpp)
      target_include_directories(test_game_compute_rtt_mirror PRIVATE tests frontends)
      target_link_libraries(test_game_compute_rtt_mirror PRIVATE prosper_live_renderer)
      add_test(NAME game_compute_exec COMMAND test_game_compute)
      add_test(NAME game_compute_rtt_destination_mirror COMMAND test_game_compute_rtt_mirror)
      add_test(NAME game_compute_rtt_destination_mirror_no_direct_retile
               COMMAND test_game_compute_rtt_mirror)
      set_tests_properties(game_compute_rtt_destination_mirror_no_direct_retile PROPERTIES
        ENVIRONMENT "PROSPER_NO_DIRECT_IMAGE_RETILE=1")
      add_test(NAME game_compute_rtt_destination_mirror_no_rgba16
               COMMAND test_game_compute_rtt_mirror)
      set_tests_properties(game_compute_rtt_destination_mirror_no_rgba16 PROPERTIES
        ENVIRONMENT "PROSPER_NO_RGBA16_COMPUTE_RTT_MIRROR=1")
      add_test(NAME game_compute_rtt_destination_mirror_no_read_alias
               COMMAND test_game_compute_rtt_mirror)
      set_tests_properties(game_compute_rtt_destination_mirror_no_read_alias PROPERTIES
        ENVIRONMENT "PROSPER_NO_COMPUTE_RTT_READ_ALIAS=1")
      add_test(NAME game_compute_rtt_destination_mirror_no_folded_read_alias
               COMMAND test_game_compute_rtt_mirror)
      set_tests_properties(game_compute_rtt_destination_mirror_no_folded_read_alias PROPERTIES
        ENVIRONMENT "PROSPER_NO_FOLDED_COMPUTE_RTT_READ_ALIAS=1")
      add_test(NAME game_compute_rtt_destination_mirror_no_one_layer_array
               COMMAND test_game_compute_rtt_mirror)
      set_tests_properties(game_compute_rtt_destination_mirror_no_one_layer_array PROPERTIES
        ENVIRONMENT "PROSPER_NO_COMPUTE_RTT_ONE_LAYER_ARRAY_DEST=1")
      add_test(NAME game_compute_exec_no_adaptive_storage_result COMMAND test_game_compute)
      set_tests_properties(game_compute_exec_no_adaptive_storage_result PROPERTIES
        ENVIRONMENT "PROSPER_NO_ADAPTIVE_STORAGE_RESULT_VALIDATION=1")
      # PROSPER_NO_SKIP_SEED forces a real seed upload; the fixture asserts that mode's own expectations
      # (identical output is rewritten) instead of the default policy's (#3390).
      add_test(NAME game_compute_exec_no_skip_seed COMMAND test_game_compute)
      set_tests_properties(game_compute_exec_no_skip_seed PROPERTIES
        ENVIRONMENT "PROSPER_NO_SKIP_SEED=1")
      # The compute path's full-surface seed/pack intermediates are pooled rather than freshly
      # allocated. Their old allocation was `make_unique_for_overwrite`, i.e. nominally
      # indeterminate -- but at these sizes it was a fresh mmap, which reads as ZERO, so a call site
      # could be depending on zeroing it never asked for. This arm hands every lease 0xCD instead:
      # it fails exactly when such a dependency exists, which is the only way that claim is testable
      # against the real backend rather than argued from the diff.
      add_test(NAME game_compute_exec_scratch_poison COMMAND test_game_compute)
      set_tests_properties(game_compute_exec_scratch_poison PROPERTIES
        ENVIRONMENT "PROSPER_DECODE_SCRATCH_POISON=1")

                     # Spells a `shared/...` include, which resolves from `frontends`.
      prosper_add_gtest(test_live_compute_descriptor_array
        SOURCES tests/shared/live/test_live_compute_descriptor_array.cpp
        INCLUDES frontends
        LIBRARIES prosper_live_renderer)

      # #3249: a dispatch result the host maps and reads needs an AVAILABILITY operation, not just a
      # fence. Registered beside the descriptor-array case because both drive the real backend
      # through execute_live_compute_items; it cannot live in the platform-independent block since
      # prosper_live_renderer exists only where Vulkan was found.
      add_executable(test_live_compute_host_read_barrier
                     tests/shared/live/test_live_compute_host_read_barrier.cpp)
      # Spells a `shared/...` include, which resolves from `frontends`.
      target_include_directories(test_live_compute_host_read_barrier PRIVATE frontends)
      target_link_libraries(test_live_compute_host_read_barrier PRIVATE prosper_live_renderer)
      add_test(NAME live_compute_host_read_barrier
               COMMAND test_live_compute_host_read_barrier)

      # #3656: device-resolved indirect dispatch -- the executor half (recording backend) and the
      # Vulkan half (the production backend consuming the argument address with
      # vkCmdDispatchIndirect). The Windows block above registers the same two sources.
      prosper_add_gtest(test_indirect_dispatch_device_route
        SOURCES tests/gpu/execute/test_indirect_dispatch_device_route.cpp
        LIBRARIES prosper_core)
      add_executable(test_live_compute_indirect_dispatch
                     tests/gpu/execute/test_live_compute_indirect_dispatch.cpp)
      target_include_directories(test_live_compute_indirect_dispatch PRIVATE tests frontends)
      target_link_libraries(test_live_compute_indirect_dispatch PRIVATE prosper_live_renderer)
      add_test(NAME live_compute_indirect_dispatch COMMAND test_live_compute_indirect_dispatch)
      set_tests_properties(live_compute_indirect_dispatch PROPERTIES TIMEOUT 120)

      # Prove our own SPIR-V emitter produces valid, correct, runnable SPIR-V (execution-differential).
      # Includes a shared fixture, which resolves from the `tests` root.
      prosper_add_gtest(test_spirv_builder
        SOURCES tests/gpu/recompiler/test_spirv_builder.cpp
        INCLUDES tests
        LIBRARIES prosper_core Vulkan::Vulkan)

      # End-to-end: recompile real RDNA2 (llvm-mc) -> SPIR-V -> run -> assert numeric result.
      add_executable(test_rdna2_to_spirv tests/gpu/recompiler/test_rdna2_to_spirv.cpp
                 tests/gpu/recompiler/test_rdna2_bvh.cpp)
      # Includes a shared fixture, which resolves from the `tests` root.
      target_include_directories(test_rdna2_to_spirv PRIVATE tests)
      target_link_libraries(test_rdna2_to_spirv PRIVATE prosper_core Vulkan::Vulkan)
      add_test(NAME rdna2_to_spirv_exec COMMAND test_rdna2_to_spirv)
      add_executable(test_scalar_ff1_b32_exec tests/gpu/execute/test_scalar_ff1_b32_exec.cpp)
      target_include_directories(test_scalar_ff1_b32_exec PRIVATE tests)
      target_link_libraries(test_scalar_ff1_b32_exec PRIVATE prosper_core Vulkan::Vulkan)
      add_test(NAME scalar_ff1_b32_exec COMMAND test_scalar_ff1_b32_exec)
      set_tests_properties(scalar_ff1_b32_exec PROPERTIES TIMEOUT 120)
      prosper_add_gtest(test_raw_register_wide_exec
        SOURCES tests/gpu/execute/test_raw_register_wide_exec.cpp
        INCLUDES tests
        LIBRARIES prosper_core Vulkan::Vulkan)
      prosper_add_gtest(test_owned_raw_x2_exec
        SOURCES tests/gpu/execute/test_owned_raw_x2_exec.cpp
        INCLUDES tests
        LIBRARIES prosper_core Vulkan::Vulkan)
      prosper_add_gtest(test_memory_fed_raw_wide_exec
        SOURCES tests/gpu/execute/test_memory_fed_raw_wide_exec.cpp
        INCLUDES tests
        LIBRARIES prosper_core Vulkan::Vulkan)
      add_executable(test_ngg_workgroup_contract
                     tests/gpu/execute/test_ngg_workgroup_contract.cpp)
      target_include_directories(test_ngg_workgroup_contract PRIVATE tests)
      target_link_libraries(test_ngg_workgroup_contract PRIVATE prosper_core Vulkan::Vulkan)
      add_test(NAME ngg_workgroup_contract COMMAND test_ngg_workgroup_contract)
      add_executable(test_portable_wave64_mask_reductions
                     tests/gpu/execute/test_portable_wave64_mask_reductions.cpp)
      target_include_directories(test_portable_wave64_mask_reductions PRIVATE tests)
      target_link_libraries(test_portable_wave64_mask_reductions PRIVATE prosper_core Vulkan::Vulkan)
      add_test(NAME portable_wave64_mask_reductions COMMAND test_portable_wave64_mask_reductions)
      set_tests_properties(portable_wave64_mask_reductions PROPERTIES TIMEOUT 60)

      add_executable(test_portable_wave64_readfirstlane
                     tests/gpu/execute/test_portable_wave64_readfirstlane.cpp)
      target_include_directories(test_portable_wave64_readfirstlane PRIVATE tests)
      target_link_libraries(test_portable_wave64_readfirstlane PRIVATE prosper_core Vulkan::Vulkan)
      add_test(NAME portable_wave64_readfirstlane COMMAND test_portable_wave64_readfirstlane)
      set_tests_properties(portable_wave64_readfirstlane PROPERTIES TIMEOUT 60)

      add_executable(test_portable_bpermute_execution tests/gpu/execute/test_portable_bpermute.cpp)
      target_include_directories(test_portable_bpermute_execution PRIVATE tests)
      target_link_libraries(test_portable_bpermute_execution PRIVATE prosper_core Vulkan::Vulkan)
      add_test(NAME portable_bpermute_execution COMMAND test_portable_bpermute_execution)
      set_tests_properties(portable_bpermute_execution PROPERTIES TIMEOUT 90)
      add_executable(ngg_capture_probe tools/ngg_probe/ngg_capture_probe.cpp)
      target_include_directories(ngg_capture_probe PRIVATE tests)
      target_link_libraries(ngg_capture_probe PRIVATE prosper_core Vulkan::Vulkan)
      add_test(NAME ngg_capture_probe_contract COMMAND ngg_capture_probe --selftest)
      add_executable(test_descriptor_array_emit tests/gpu/test_descriptor_array_emit.cpp)
      target_link_libraries(test_descriptor_array_emit PRIVATE prosper_core Vulkan::Vulkan)
      add_test(NAME descriptor_array_emit COMMAND test_descriptor_array_emit)


      # The DS layer-stride registry's identity (base+width+height, not base alone) and
      # PersistentDsKey's field membership.
      # Includes a shared fixture, which resolves from the `tests` root.
      # Reaches `shared/...` through render_runner.h, which resolves from `frontends`.
      prosper_add_gtest(test_ds_layer_stride
        SOURCES tests/render/test_ds_layer_stride.cpp
        INCLUDES tests frontends
        LIBRARIES prosper_core Vulkan::Vulkan)

      # The per-draw internal-GDS scan memo (#2334). Two registrations of ONE binary: the second
      # sets the opt-out, so the pair proves the memo both fires and can be turned off, on a single
      # build. Both must PASS -- the binary flips its own expectation from the environment.
      # Includes a shared fixture, which resolves from the `tests` root.
      # Reaches `shared/...` through render_runner.h, which resolves from `frontends`.
      prosper_add_gtest(test_gds_scan_memo
        SOURCES tests/render/test_gds_scan_memo.cpp
        INCLUDES tests frontends
        LIBRARIES prosper_core Vulkan::Vulkan)
      add_test(NAME gds_scan_memo_disabled COMMAND test_gds_scan_memo)
      set_tests_properties(gds_scan_memo_disabled PROPERTIES
                           ENVIRONMENT "PROSPER_NO_GDS_SCAN_MEMO=1")

      # The backend's persistent-resource domain lock (#2953). Carries its own positive control --
      # an unguarded arm that proves the overlap detector fires at all -- so a clean zero in the
      # guarded arms cannot be read as "the threads never overlapped".
      add_executable(test_backend_persistent_resource_lock
                     tests/render/test_backend_persistent_resource_lock.cpp)
      # Includes a shared fixture, which resolves from the `tests` root.
      target_include_directories(test_backend_persistent_resource_lock PRIVATE tests)
      # Reaches `shared/...` through render_runner.h, which resolves from `frontends`.
      target_include_directories(test_backend_persistent_resource_lock PRIVATE frontends)
      target_link_libraries(test_backend_persistent_resource_lock
                            PRIVATE prosper_core Vulkan::Vulkan)
      add_test(NAME backend_persistent_resource_lock
               COMMAND test_backend_persistent_resource_lock)

      # #2944: every GPU readback in the render backend is host-read after a fence wait, and a fence
      # orders execution rather than performing the availability operation into the host domain.
      # Structural by necessity -- on a coherent desktop driver the UNFIXED code returns correct
      # pixels, so the assertion is that the barrier is RECORDED, with a no-readback negative control
      # so the counter cannot be satisfied by any render at all. Same include roots and same real
      # Vulkan device as the render_runner.h tests above.
      # Includes a shared fixture, which resolves from the `tests` root.
      # Reaches `shared/...` through render_runner.h, which resolves from `frontends`.
      prosper_add_gtest(test_host_read_barrier
        SOURCES tests/render/test_host_read_barrier.cpp
        INCLUDES tests frontends
        LIBRARIES prosper_core Vulkan::Vulkan)

      # #3248: the mip-assembly path cleared the destination image and then copied into it with no
      # memory dependency between the two writes. Structural for a sharper reason than the readback
      # above -- assembly clears to BLACK deliberately, so a clear that lands after a copy produces
      # exactly what a correct run produces for a missing level. No content assertion can separate
      # them; the negative control is an ordinary upload, which has no clear-then-copy pair.
      # Includes a shared fixture, which resolves from the `tests` root.
      # Reaches `shared/...` through render_runner.h, which resolves from `frontends`.
      prosper_add_gtest(test_mip_assembly_barrier
        SOURCES tests/render/test_mip_assembly_barrier.cpp
        INCLUDES tests frontends
        LIBRARIES prosper_core Vulkan::Vulkan)

      # #3248: PROSPER_GEOM_PROBE and PROSPER_DRAW_ISO record Vulkan commands that no test executed,
      # so tools/vkval could not see them however it was configured. This case runs both once. Its
      # own assertions pin the arming decisions; the SPEC misuse they used to commit is caught by
      # running the suite under the validation layer, which is only possible because this exists.
      add_executable(test_render_diagnostic_paths tests/render/test_render_diagnostic_paths.cpp)
      # Includes a shared fixture, which resolves from the `tests` root.
      target_include_directories(test_render_diagnostic_paths PRIVATE tests)
      # Reaches `shared/...` through render_runner.h, which resolves from `frontends`.
      target_include_directories(test_render_diagnostic_paths PRIVATE frontends)
      target_link_libraries(test_render_diagnostic_paths PRIVATE prosper_core Vulkan::Vulkan)
      add_test(NAME render_diagnostic_paths COMMAND test_render_diagnostic_paths)
      # Armed HERE rather than with setenv() inside the test. PROSPER_GEOM_PROBE is a cached
      # one-shot read (PROSPER_ENV_VALUE), so a test that sets it after something has read it arms
      # nothing while still passing -- which `cached_env_arming_logic` refuses by name. The test
      # ENVIRONMENT is in place before the process starts, so read order cannot make it vacuous, and
      # the test fails loudly when run without it. PROSPER_ISO_AT must name a pixel inside the
      # 32x32 frame the test renders.
      set_tests_properties(render_diagnostic_paths PROPERTIES ENVIRONMENT
                           "PROSPER_GEOM_PROBE=0;PROSPER_DRAW_ISO=1;PROSPER_ISO_AT=16,16;PROSPER_FRAME_DIR=render_diagnostic_paths_frames")

      # Exercise every opt-in depth-clear policy under Vulkan validation. The test asserts that
      # the selected draw reached its path, so a clean layer log cannot be a vacuous pass.
      foreach(clear_probe_mode IN ITEMS 1 2 3 4 5)
        add_test(NAME render_diagnostic_paths_clear_${clear_probe_mode}
                 COMMAND test_render_diagnostic_paths ${clear_probe_mode})
        set_tests_properties(render_diagnostic_paths_clear_${clear_probe_mode} PROPERTIES ENVIRONMENT
                             "PROSPER_GEOM_PROBE=0;PROSPER_DRAW_ISO=1;PROSPER_ISO_AT=16,16;PROSPER_FRAME_DIR=render_diagnostic_paths_frames_${clear_probe_mode};PROSPER_DIAG_CLEAR_OVERWRITE=${clear_probe_mode}")
      endforeach()

      # The address-watch matcher's edge cases (exclusive end, empty range, UINT64_MAX wrap).
      prosper_add_gtest(test_compute_address_watch
        SOURCES tests/gpu/test_compute_address_watch.cpp
        LIBRARIES prosper_core Vulkan::Vulkan)

      # The EXEC-driven pointer-chasing loop shape GTA V hangs on, plus PROSPER_CFG_TRIP_BOUND.
      add_executable(test_cfg_trip_bound tests/shared/diagnostics/test_cfg_trip_bound.cpp)
      # Includes a shared fixture, which resolves from the `tests` root.
      target_include_directories(test_cfg_trip_bound PRIVATE tests)
      # Spells a `shared/...` include, which resolves from `frontends`.
      target_include_directories(test_cfg_trip_bound PRIVATE frontends)
      target_link_libraries(test_cfg_trip_bound PRIVATE prosper_core Vulkan::Vulkan)
      add_test(NAME cfg_trip_bound_exec COMMAND test_cfg_trip_bound)

      # The same loop shape, but CHASING A REAL BUFFER through a V# -- the half test_cfg_trip_bound
      # deliberately leaves out ("decrements the index instead of chasing a buffer").
      prosper_add_gtest(test_traversal_chase
        SOURCES tests/gpu/recompiler/test_traversal_chase.cpp
        INCLUDES tests
        LIBRARIES prosper_core Vulkan::Vulkan)

      # Render a triangle whose PIXEL shader is recompiled from RDNA2 (EXP MRT0 -> color output).
      # Includes a shared fixture, which resolves from the `tests` root.
      # Reaches `shared/...` through render_runner.h, which resolves from `frontends`.
      prosper_add_gtest(test_recompiled_fragment
        SOURCES tests/gpu/recompiler/test_recompiled_fragment.cpp
        INCLUDES tests frontends
        LIBRARIES prosper_core Vulkan::Vulkan)

      add_executable(test_fragment_loop_mask_exec tests/gpu/execute/test_fragment_loop_mask_exec.cpp)
      target_include_directories(test_fragment_loop_mask_exec PRIVATE tests frontends)
      target_link_libraries(test_fragment_loop_mask_exec PRIVATE prosper_core Vulkan::Vulkan)
      add_test(NAME fragment_loop_mask_exec COMMAND test_fragment_loop_mask_exec)
      set_tests_properties(fragment_loop_mask_exec PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 60)

      # A Wave64 lane-id/ballot/MBCNT/vote pixel shader executed on this device's own fragment
      # subgroup, narrower than the guest wave included (#3464's partial-wave tier). Checks are
      # invariant under any grouping of pixels into waves.
      add_executable(test_fragment_partial_wave_exec tests/gpu/execute/test_fragment_partial_wave_exec.cpp)
      target_include_directories(test_fragment_partial_wave_exec PRIVATE tests frontends)
      target_link_libraries(test_fragment_partial_wave_exec PRIVATE prosper_core Vulkan::Vulkan)
      add_test(NAME fragment_partial_wave_exec COMMAND test_fragment_partial_wave_exec)
      set_tests_properties(fragment_partial_wave_exec PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 60)

      add_executable(test_fragment_numeric_mbcnt tests/gpu/execute/test_fragment_numeric_mbcnt.cpp)
      target_include_directories(test_fragment_numeric_mbcnt PRIVATE tests frontends)
      target_link_libraries(test_fragment_numeric_mbcnt PRIVATE prosper_core Vulkan::Vulkan)
      add_test(NAME fragment_numeric_mbcnt COMMAND test_fragment_numeric_mbcnt)
      set_tests_properties(fragment_numeric_mbcnt PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 60)

      add_executable(test_fragment_proven_votes tests/gpu/execute/test_fragment_proven_votes.cpp)
      target_include_directories(test_fragment_proven_votes PRIVATE tests frontends)
      target_link_libraries(test_fragment_proven_votes PRIVATE prosper_core Vulkan::Vulkan)
      add_executable(test_fragment_vote_refusal_reporting tests/gpu/execute/test_fragment_vote_refusal_reporting.cpp)
      target_include_directories(test_fragment_vote_refusal_reporting PRIVATE tests frontends)
      target_link_libraries(test_fragment_vote_refusal_reporting PRIVATE prosper_core Vulkan::Vulkan)
      add_test(NAME fragment_vote_refusal_reporting COMMAND test_fragment_vote_refusal_reporting)
      set_tests_properties(fragment_vote_refusal_reporting PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 60)
      add_test(NAME fragment_proven_votes COMMAND test_fragment_proven_votes)
      set_tests_properties(fragment_proven_votes PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 60)
      add_test(NAME fragment_proven_votes_no_robust2 COMMAND test_fragment_proven_votes --expect-no-robust2)
      set_tests_properties(fragment_proven_votes_no_robust2 PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 60
        ENVIRONMENT "PROSPER_NO_ROBUST_BUFFER_ACCESS2=1")

      add_executable(test_numeric_mbcnt tests/gpu/execute/test_numeric_mbcnt.cpp)
      target_include_directories(test_numeric_mbcnt PRIVATE tests)
      target_link_libraries(test_numeric_mbcnt PRIVATE prosper_core Vulkan::Vulkan)
      add_test(NAME numeric_mbcnt COMMAND test_numeric_mbcnt)
      set_tests_properties(numeric_mbcnt PROPERTIES TIMEOUT 60)

      add_executable(test_numeric_wqm tests/gpu/execute/test_numeric_wqm.cpp)
      target_include_directories(test_numeric_wqm PRIVATE tests)
      target_link_libraries(test_numeric_wqm PRIVATE prosper_core Vulkan::Vulkan)
      add_test(NAME numeric_wqm COMMAND test_numeric_wqm)
      set_tests_properties(numeric_wqm PROPERTIES TIMEOUT 90)

      # Render a frame where BOTH vertex + pixel shaders are recompiled from RDNA2.
      # Includes a shared fixture, which resolves from the `tests` root.
      # Reaches `shared/...` through render_runner.h, which resolves from `frontends`.
      prosper_add_gtest(test_recompiled_shaders
        SOURCES tests/gpu/recompiler/test_recompiled_shaders.cpp
        INCLUDES tests frontends
        LIBRARIES prosper_core Vulkan::Vulkan)

      # Host-side workgroup/mesh execution boundary for merged NGG. On devices without the
      # optional extension the same fixture asserts that the draw leaves the clear image intact.
      add_executable(test_mesh_backend tests/gpu/recompiler/test_mesh_backend.cpp)
      target_include_directories(test_mesh_backend PRIVATE tests frontends)
      target_link_libraries(test_mesh_backend PRIVATE prosper_core Vulkan::Vulkan)
      add_test(NAME mesh_backend_workgroup COMMAND test_mesh_backend)
      set_tests_properties(mesh_backend_workgroup PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 60)
      add_test(NAME mesh_backend_feature_off COMMAND test_mesh_backend)
      set_tests_properties(mesh_backend_feature_off PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 60
        ENVIRONMENT "PROSPER_NO_MESH_SHADER=1")

      add_executable(test_volume_target_handoff tests/gpu/execute/test_volume_target_handoff.cpp)
      target_include_directories(test_volume_target_handoff PRIVATE tests frontends)
      target_link_libraries(test_volume_target_handoff PRIVATE prosper_core Vulkan::Vulkan)
      add_test(NAME volume_target_handoff COMMAND test_volume_target_handoff)
      set_tests_properties(volume_target_handoff PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 60)

      # Vertex fetch: a VS that pulls positions from a bound vertex buffer via buffer_load_format,
      # keyed by a ShaderResourceTable — real game vertex shaders work this way.
      # Includes a shared fixture, which resolves from the `tests` root.
      # Reaches `shared/...` through render_runner.h, which resolves from `frontends`.
      prosper_add_gtest(test_vertex_fetch_render
        SOURCES tests/gpu/test_vertex_fetch_render.cpp
        INCLUDES tests frontends
        LIBRARIES prosper_core Vulkan::Vulkan)

      # Texture sampling: a recompiled pixel shader does MIMG image_sample against a bound 2x2 texture.
      add_executable(test_texture_sample_render tests/gpu/test_texture_sample_render.cpp)
      add_executable(test_float32_array_gather tests/gpu/execute/test_float32_array_gather.cpp)
      target_include_directories(test_float32_array_gather PRIVATE tests frontends)
      target_link_libraries(test_float32_array_gather PRIVATE prosper_core Vulkan::Vulkan)
      add_test(NAME float32_array_gather_compile COMMAND test_float32_array_gather --compile-only)
      add_test(NAME float32_array_gather COMMAND test_float32_array_gather)
      # Includes a shared fixture, which resolves from the `tests` root.
      target_include_directories(test_texture_sample_render PRIVATE tests)
      # Reaches `shared/...` through render_runner.h, which resolves from `frontends`.
      target_include_directories(test_texture_sample_render PRIVATE frontends)
      target_link_libraries(test_texture_sample_render PRIVATE prosper_core Vulkan::Vulkan)
      add_test(NAME texture_sample_render COMMAND ${CMAKE_COMMAND} -E env
          --unset=PROSPER_BACKEND_TEXTURE_CACHE_MB $<TARGET_FILE:test_texture_sample_render>)
      add_test(NAME backend_texture_path_census_live COMMAND ${CMAKE_COMMAND} -E env
          --unset=PROSPER_BACKEND_TEXTURE_CACHE_MB
          --unset=PROSPER_BACKEND_TEXTURE_PATH_CENSUS_MIN_DRAWS
          PROSPER_BACKEND_TEXTURE_PATH_CENSUS=1 $<TARGET_FILE:test_texture_sample_render>)
      add_test(NAME backend_texture_pressure_flush COMMAND ${CMAKE_COMMAND} -E env
          --unset=PROSPER_BACKEND_TEXTURE_CACHE_MB
          PROSPER_BACKEND_TEXTURE_PRESSURE_FLUSH=1
          PROSPER_RENDER_TIMING=1
          $<TARGET_FILE:test_texture_sample_render> --cache-pressure)
      add_test(NAME backend_texture_pressure_control COMMAND ${CMAKE_COMMAND} -E env
          --unset=PROSPER_BACKEND_TEXTURE_PRESSURE_FLUSH
          --unset=PROSPER_BACKEND_TEXTURE_CACHE_MB
          PROSPER_RENDER_TIMING=1
          $<TARGET_FILE:test_texture_sample_render> --cache-pressure)
      # #3873: a full resident-texture cache admits a new image while an ordered batch is pending,
      # retiring the idle victim through the batch completion. The opt-out arm pins the old skip.
      add_test(NAME backend_texture_deferred_eviction COMMAND ${CMAKE_COMMAND} -E env
          --unset=PROSPER_BACKEND_TEXTURE_CACHE_MB
          --unset=PROSPER_BACKEND_TEXTURE_PRESSURE_FLUSH
          --unset=PROSPER_NO_DEFERRED_TEXTURE_EVICTION
          $<TARGET_FILE:test_texture_sample_render> --deferred-eviction)
      add_test(NAME backend_texture_deferred_eviction_optout COMMAND ${CMAKE_COMMAND} -E env
          --unset=PROSPER_BACKEND_TEXTURE_CACHE_MB
          --unset=PROSPER_BACKEND_TEXTURE_PRESSURE_FLUSH
          PROSPER_NO_DEFERRED_TEXTURE_EVICTION=1
          $<TARGET_FILE:test_texture_sample_render> --deferred-eviction)

      # Multi-draw backend: N draws composited into ONE cleared-once framebuffer (red opaque + green
      # additive -> yellow center proves accumulation into a single target).
      # Includes a shared fixture, which resolves from the `tests` root.
      # Reaches `shared/...` through render_runner.h, which resolves from `frontends`.
      prosper_add_gtest(test_multidraw_render
        SOURCES tests/gpu/state/test_multidraw_render.cpp
        INCLUDES tests frontends
        LIBRARIES prosper_core Vulkan::Vulkan)

      prosper_add_gtest(test_renderer_uint_copy
        SOURCES tests/gpu/execute/test_renderer_uint_copy.cpp
        INCLUDES tests frontends
        LIBRARIES prosper_live_renderer prosper_core Vulkan::Vulkan)

      # #4291: BGRA render targets reach compute in guest component order on every path.
      prosper_add_gtest(test_live_compute_bgra_target
        SOURCES tests/shared/live/test_live_compute_bgra_target.cpp
        INCLUDES tests frontends src
        LIBRARIES prosper_live_renderer prosper_core Vulkan::Vulkan)

      add_executable(test_storage_metadata_borrow tests/gpu/execute/test_storage_metadata_borrow.cpp)
      target_include_directories(test_storage_metadata_borrow PRIVATE tests frontends)
      target_link_libraries(test_storage_metadata_borrow PRIVATE prosper_live_renderer prosper_core Vulkan::Vulkan)
      add_test(NAME storage_metadata_borrow COMMAND ${CMAKE_COMMAND} -E env
               PROSPER_COMPUTE_IMAGE_CACHE_MIN_KB=1 PROSPER_COMPUTE_STORAGE_IMAGE_CACHE_MIN_KB=1
               $<TARGET_FILE:test_storage_metadata_borrow>)
      set_tests_properties(storage_metadata_borrow PROPERTIES TIMEOUT 60)

      add_executable(test_storage_output_conflicts tests/gpu/execute/test_storage_output_conflicts.cpp)
      target_include_directories(test_storage_output_conflicts PRIVATE tests frontends)
      target_link_libraries(test_storage_output_conflicts PRIVATE prosper_live_renderer prosper_core Vulkan::Vulkan)
      add_test(NAME storage_output_conflicts COMMAND ${CMAKE_COMMAND} -E env
               PROSPER_COMPUTE_IMAGE_CACHE_MIN_KB=1 PROSPER_COMPUTE_STORAGE_IMAGE_CACHE_MIN_KB=1 PROSPER_COMPUTE_BUFFER_RESULT_MIN_MB=1 $<TARGET_FILE:test_storage_output_conflicts>)
      add_test(NAME storage_output_conflicts_unjournaled COMMAND ${CMAKE_COMMAND} -E env
               PROSPER_COMPUTE_IMAGE_CACHE_MIN_KB=1 PROSPER_COMPUTE_STORAGE_IMAGE_CACHE_MIN_KB=1 PROSPER_COMPUTE_BUFFER_RESULT_MIN_MB=1 $<TARGET_FILE:test_storage_output_conflicts> --unjournaled)
      add_test(NAME storage_output_conflicts_overflow COMMAND ${CMAKE_COMMAND} -E env
               PROSPER_COMPUTE_IMAGE_CACHE_MIN_KB=1 PROSPER_COMPUTE_STORAGE_IMAGE_CACHE_MIN_KB=1 PROSPER_COMPUTE_BUFFER_RESULT_MIN_MB=1 $<TARGET_FILE:test_storage_output_conflicts> --overflow)
      add_test(NAME storage_output_conflicts_submit_failure COMMAND ${CMAKE_COMMAND} -E env
               PROSPER_COMPUTE_IMAGE_CACHE_MIN_KB=1 PROSPER_COMPUTE_STORAGE_IMAGE_CACHE_MIN_KB=1
               $<TARGET_FILE:test_storage_output_conflicts> --submit-failure)
      if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        add_test(NAME storage_output_conflicts_watched COMMAND ${CMAKE_COMMAND} -E env
                 PROSPER_COMPUTE_IMAGE_CACHE_MIN_KB=1 PROSPER_COMPUTE_STORAGE_IMAGE_CACHE_MIN_KB=1
                 $<TARGET_FILE:test_storage_output_conflicts> --watched)
        set_tests_properties(storage_output_conflicts_watched PROPERTIES TIMEOUT 60)
        add_executable(test_cached_compute_fill tests/gpu/execute/test_cached_compute_fill.cpp)
        target_include_directories(test_cached_compute_fill PRIVATE tests frontends)
        target_link_libraries(test_cached_compute_fill PRIVATE prosper_live_renderer prosper_core Vulkan::Vulkan)
        add_test(NAME cached_compute_fill COMMAND ${CMAKE_COMMAND} -E env
                 PROSPER_COMPUTE_WRITE_WATCH_PROMOTE_HITS=1
                 PROSPER_COMPUTE_BUFFER_RESULT_MIN_MB=0
                 $<TARGET_FILE:test_cached_compute_fill>)
        add_test(NAME cached_compute_fill_override COMMAND ${CMAKE_COMMAND} -E env
                 PROSPER_COMPUTE_WRITE_WATCH_PROMOTE_HITS=1
                 PROSPER_COMPUTE_BUFFER_RESULT_MIN_MB=0
                 $<TARGET_FILE:test_cached_compute_fill> --cached-override)
        set_tests_properties(cached_compute_fill cached_compute_fill_override PROPERTIES TIMEOUT 60)
      endif()
      set_tests_properties(storage_output_conflicts storage_output_conflicts_unjournaled
                           storage_output_conflicts_overflow storage_output_conflicts_submit_failure PROPERTIES TIMEOUT 60)

      add_executable(test_storage_readonly tests/gpu/execute/test_storage_readonly.cpp)
      target_include_directories(test_storage_readonly PRIVATE tests frontends)
      target_link_libraries(test_storage_readonly PRIVATE prosper_live_renderer prosper_core Vulkan::Vulkan)
      add_test(NAME storage_readonly COMMAND ${CMAKE_COMMAND} -E env
               --unset=PROSPER_NO_READONLY_STORAGE_WRITEBACK_SKIP
               PROSPER_COMPUTE_STORAGE_IMAGE_CACHE_MIN_KB=1
               $<TARGET_FILE:test_storage_readonly>)
      add_test(NAME storage_readonly_control COMMAND ${CMAKE_COMMAND} -E env
               PROSPER_NO_READONLY_STORAGE_WRITEBACK_SKIP=1
               PROSPER_COMPUTE_STORAGE_IMAGE_CACHE_MIN_KB=1
               $<TARGET_FILE:test_storage_readonly> --control)
      set_tests_properties(storage_readonly storage_readonly_control PROPERTIES TIMEOUT 60)

      add_executable(test_storage_rtt_gpu_seed tests/gpu/execute/test_storage_rtt_gpu_seed.cpp)
      target_include_directories(test_storage_rtt_gpu_seed PRIVATE tests frontends)
      target_link_libraries(test_storage_rtt_gpu_seed PRIVATE prosper_live_renderer prosper_core Vulkan::Vulkan)
      add_test(NAME storage_rtt_gpu_seed COMMAND ${CMAKE_COMMAND} -E env
               --unset=PROSPER_NO_STANDALONE_RTT_SEED $<TARGET_FILE:test_storage_rtt_gpu_seed>)
      add_test(NAME storage_rtt_cpu_seed COMMAND ${CMAKE_COMMAND} -E env
               PROSPER_NO_STANDALONE_RTT_SEED=1 $<TARGET_FILE:test_storage_rtt_gpu_seed> --cpu)
      add_test(NAME storage_rtt_result_retention COMMAND ${CMAKE_COMMAND} -E env
               --unset=PROSPER_NO_STANDALONE_RTT_SEED
               --unset=PROSPER_NO_RENDERER_SEEDED_RESULT_CACHE
               $<TARGET_FILE:test_storage_rtt_gpu_seed> --retention)
      add_test(NAME storage_rtt_result_retention_off COMMAND ${CMAKE_COMMAND} -E env
               --unset=PROSPER_NO_STANDALONE_RTT_SEED
               PROSPER_NO_RENDERER_SEEDED_RESULT_CACHE=1
               $<TARGET_FILE:test_storage_rtt_gpu_seed> --retention-off)
      set_tests_properties(storage_rtt_gpu_seed storage_rtt_cpu_seed
                           storage_rtt_result_retention storage_rtt_result_retention_off
                           PROPERTIES TIMEOUT 60)

      add_executable(test_compute_transfer_mip_chain
                     tests/gpu/execute/test_compute_transfer_mip_chain.cpp)
      target_include_directories(test_compute_transfer_mip_chain PRIVATE tests frontends)
      target_link_libraries(test_compute_transfer_mip_chain PRIVATE
                           prosper_live_renderer prosper_core Vulkan::Vulkan)
      add_test(NAME compute_transfer_mip_chain COMMAND ${CMAKE_COMMAND} -E env
               PROSPER_COMPUTE_IMAGE_CACHE_MIN_KB=0
               PROSPER_COMPUTE_STORAGE_IMAGE_CACHE_MIN_KB=0
               $<TARGET_FILE:test_compute_transfer_mip_chain>)
      set_tests_properties(compute_transfer_mip_chain PROPERTIES TIMEOUT 60)

      add_executable(test_near_full_storage_authority
                     tests/gpu/execute/test_near_full_storage_authority.cpp)
      target_include_directories(test_near_full_storage_authority PRIVATE tests frontends)
      target_link_libraries(test_near_full_storage_authority PRIVATE
                           prosper_live_renderer prosper_core Vulkan::Vulkan)
      add_test(NAME near_full_storage_authority COMMAND ${CMAKE_COMMAND} -E env
               --unset=PROSPER_SKIP_EXPORTED_STORAGE_WRITEBACK
               $<TARGET_FILE:test_near_full_storage_authority>)
      add_test(NAME near_full_storage_authority_export COMMAND ${CMAKE_COMMAND} -E env
               PROSPER_SKIP_EXPORTED_STORAGE_WRITEBACK=1
               $<TARGET_FILE:test_near_full_storage_authority>)
      add_test(NAME near_full_storage_full_transition_default COMMAND ${CMAKE_COMMAND} -E env
               --unset=PROSPER_SKIP_EXPORTED_STORAGE_WRITEBACK
               $<TARGET_FILE:test_near_full_storage_authority> full-transition)
      add_test(NAME near_full_storage_full_transition COMMAND ${CMAKE_COMMAND} -E env
               PROSPER_SKIP_EXPORTED_STORAGE_WRITEBACK=1
               $<TARGET_FILE:test_near_full_storage_authority> full-transition)
      add_test(NAME storage_seed_program_replacement COMMAND ${CMAKE_COMMAND} -E env
               --unset=PROSPER_NO_SKIP_SEED
               $<TARGET_FILE:test_near_full_storage_authority> program-replacement)
      add_test(NAME storage_seed_program_replacement_seeded COMMAND ${CMAKE_COMMAND} -E env
               PROSPER_NO_SKIP_SEED=1
               $<TARGET_FILE:test_near_full_storage_authority> program-replacement)
      add_executable(test_storage_float_roundtrip tests/shared/live/test_storage_float_roundtrip.cpp)
      target_include_directories(test_storage_float_roundtrip PRIVATE frontends)
      target_link_libraries(test_storage_float_roundtrip PRIVATE prosper_live_renderer prosper_core Vulkan::Vulkan)
      foreach(unpack 0 1)
        foreach(pack 0 1)
          set(unpack_env --unset=PROSPER_NO_F16C_STORAGE_UNPACK)
          set(pack_env --unset=PROSPER_NO_F16C_STORAGE_PACK)
          if(unpack)
            set(unpack_env PROSPER_NO_F16C_STORAGE_UNPACK=1)
          endif()
          if(pack)
            set(pack_env PROSPER_NO_F16C_STORAGE_PACK=1)
          endif()
          add_test(NAME storage_float_roundtrip_${unpack}_${pack} COMMAND ${CMAKE_COMMAND} -E env
                   ${unpack_env} ${pack_env}
                   $<TARGET_FILE:test_storage_float_roundtrip>)
        endforeach()
      endforeach()
      add_executable(test_storage_raw_residency tests/gpu/execute/test_storage_raw_residency.cpp)
      target_include_directories(test_storage_raw_residency PRIVATE tests frontends)
      target_link_libraries(test_storage_raw_residency PRIVATE prosper_live_renderer prosper_core Vulkan::Vulkan)
      add_test(NAME storage_raw_residency_seeded COMMAND ${CMAKE_COMMAND} -E env
               PROSPER_NO_SKIP_SEED=1 $<TARGET_FILE:test_storage_raw_residency>)
      if(Python3_Interpreter_FOUND)
        add_test(NAME storage_raw_residency COMMAND "${Python3_EXECUTABLE}"
                 "${CMAKE_SOURCE_DIR}/tests/gpu/execute/test_storage_raw_residency.py"
                 $<TARGET_FILE:test_storage_raw_residency>)
      endif()
      add_executable(test_storage_snorm_untouched tests/gpu/execute/test_storage_snorm_untouched.cpp)
      target_include_directories(test_storage_snorm_untouched PRIVATE tests frontends)
      target_link_libraries(test_storage_snorm_untouched PRIVATE prosper_live_renderer prosper_core Vulkan::Vulkan)
      foreach(bits 8 16)
        add_test(NAME storage_snorm_${bits}_aliases COMMAND test_storage_snorm_untouched ${bits} partial aliases)
        set_tests_properties(storage_snorm_${bits}_aliases PROPERTIES TIMEOUT 60)
        foreach(mode none partial full)
          add_test(NAME storage_snorm_${bits}_${mode} COMMAND test_storage_snorm_untouched ${bits} ${mode})
          set_tests_properties(storage_snorm_${bits}_${mode} PROPERTIES TIMEOUT 60)
        endforeach()
      endforeach()
      add_executable(test_seed_coverage_dynamic tests/gpu/execute/test_seed_coverage_dynamic.cpp)
      target_include_directories(test_seed_coverage_dynamic PRIVATE tests frontends)
      target_link_libraries(test_seed_coverage_dynamic PRIVATE
                           prosper_live_renderer prosper_core Vulkan::Vulkan)
      add_test(NAME storage_seed_launch COMMAND ${CMAKE_COMMAND} -E env
               --unset=PROSPER_NO_SKIP_SEED $<TARGET_FILE:test_seed_coverage_dynamic> launch)
      add_test(NAME storage_seed_launch_seeded COMMAND ${CMAKE_COMMAND} -E env
               PROSPER_NO_SKIP_SEED=1 $<TARGET_FILE:test_seed_coverage_dynamic> launch)
      foreach(mode mask coordinates layers)
        add_test(NAME storage_seed_${mode} COMMAND ${CMAKE_COMMAND} -E env
                 --unset=PROSPER_NO_SKIP_SEED $<TARGET_FILE:test_seed_coverage_dynamic> ${mode})
        add_test(NAME storage_seed_${mode}_seeded COMMAND ${CMAKE_COMMAND} -E env
                 PROSPER_NO_SKIP_SEED=1 $<TARGET_FILE:test_seed_coverage_dynamic> ${mode})
        set_tests_properties(storage_seed_${mode} storage_seed_${mode}_seeded PROPERTIES TIMEOUT 60)
      endforeach()
      set_tests_properties(storage_seed_launch storage_seed_launch_seeded
                           storage_seed_mask_seeded PROPERTIES TIMEOUT 60)
      if(Python3_Interpreter_FOUND)
        add_test(NAME storage_seed_program_reuse
                 COMMAND "${Python3_EXECUTABLE}"
                         "${CMAKE_SOURCE_DIR}/tests/gpu/execute/test_storage_seed_proof_reuse.py"
                         2 $<TARGET_FILE:test_near_full_storage_authority> program-replacement)
        add_test(NAME storage_seed_variant_eviction
                 COMMAND "${Python3_EXECUTABLE}"
                         "${CMAKE_SOURCE_DIR}/tests/gpu/execute/test_storage_seed_proof_reuse.py"
                         4 $<TARGET_FILE:test_near_full_storage_authority> variant-eviction)
        add_test(NAME storage_seed_launch_reuse
                 COMMAND "${Python3_EXECUTABLE}"
                         "${CMAKE_SOURCE_DIR}/tests/gpu/execute/test_storage_seed_proof_reuse.py"
                         1 $<TARGET_FILE:test_seed_coverage_dynamic> launch)
        set_tests_properties(storage_seed_program_reuse storage_seed_launch_reuse
                             storage_seed_variant_eviction
                             PROPERTIES TIMEOUT 60)
      endif()
      set_tests_properties(near_full_storage_authority near_full_storage_authority_export
                           near_full_storage_full_transition
                           near_full_storage_full_transition_default
                           storage_seed_program_replacement storage_seed_program_replacement_seeded
                           PROPERTIES TIMEOUT 60)

      if(Python3_Interpreter_FOUND)
        add_executable(test_compute_buffer_timing tests/gpu/execute/test_compute_buffer_timing.cpp)
        target_include_directories(test_compute_buffer_timing PRIVATE tests frontends)
        target_link_libraries(test_compute_buffer_timing PRIVATE
                             prosper_live_renderer prosper_core Vulkan::Vulkan)
        add_test(NAME compute_buffer_timing_runtime
                 COMMAND "${Python3_EXECUTABLE}"
                         "${CMAKE_SOURCE_DIR}/tools/perf/test_compute_buffer_timing_runtime.py"
                         $<TARGET_FILE:test_compute_buffer_timing>)
        set_tests_properties(compute_buffer_timing_runtime PROPERTIES TIMEOUT 180)
        add_test(NAME compute_buffer_hosted_overlap
                 COMMAND test_compute_buffer_timing hosted-overlap "${CMAKE_CURRENT_BINARY_DIR}")
        set_tests_properties(compute_buffer_hosted_overlap PROPERTIES TIMEOUT 120)
        # #3425: the compute driver pipeline cache is written on the frontend's _Exit path, not by a
        # destructor. Separate processes are required -- a single process could pass while writing
        # nothing -- and the driver also asserts prosper-app still names the flush.
        add_executable(test_compute_cache_persistence tests/gpu/execute/test_compute_cache_persistence.cpp)
        target_include_directories(test_compute_cache_persistence PRIVATE tests frontends)
        target_link_libraries(test_compute_cache_persistence PRIVATE
                             prosper_live_renderer prosper_core Vulkan::Vulkan)
        add_test(NAME compute_cache_persistence
                 COMMAND "${Python3_EXECUTABLE}"
                         "${CMAKE_CURRENT_SOURCE_DIR}/tests/gpu/execute/test_compute_cache_persistence.py"
                         $<TARGET_FILE:test_compute_cache_persistence>
                         "${CMAKE_CURRENT_SOURCE_DIR}/frontends/prosper-app/main.cpp")
        set_tests_properties(compute_cache_persistence PROPERTIES TIMEOUT 300)

        add_executable(test_compute_buffer_residency tests/gpu/execute/test_compute_buffer_residency.cpp)
        target_include_directories(test_compute_buffer_residency PRIVATE tests frontends)
        target_link_libraries(test_compute_buffer_residency PRIVATE
                             prosper_live_renderer prosper_core Vulkan::Vulkan)
        add_test(NAME compute_buffer_residency_runtime
                 COMMAND "${Python3_EXECUTABLE}"
                         "${CMAKE_SOURCE_DIR}/tools/perf/test_compute_buffer_residency_runtime.py"
                         $<TARGET_FILE:test_compute_buffer_residency>)
        set_tests_properties(compute_buffer_residency_runtime PROPERTIES TIMEOUT 180)
      endif()

      # #3180: a failed color-target vkCreateImage must name its site and the real VkResult. Same
      # block as the other render_runner.h tests -- it needs the `tests` and `frontends` include
      # roots and a real Vulkan device.
      prosper_add_gtest(test_color_target_create_failure
        SOURCES tests/gpu/test_color_target_create_failure.cpp
        INCLUDES tests frontends
        LIBRARIES prosper_core Vulkan::Vulkan)

      # #3210: the render pass, framebuffer, image-view, sampler and shader-module creates in the
      # same pass. Two of them wrote UNINITIALIZED handles that were then used unchecked. Same
      # block and same reason as the test above -- `tests` and `frontends` include roots plus a
      # real Vulkan device.
      prosper_add_gtest(test_render_object_create_failure
        SOURCES tests/gpu/test_render_object_create_failure.cpp
        INCLUDES tests frontends
        LIBRARIES prosper_core Vulkan::Vulkan)

      # Indexed draws (#64): vkCmdDrawIndexed from a BackendDraw's index data — the [0,1,2,2,3,0] quad
      # must match the retired perimeter-fan rendering pixel-exact, and indices must really be applied.
      # Includes a shared fixture, which resolves from the `tests` root.
      # Reaches `shared/...` through render_runner.h, which resolves from `frontends`.
      prosper_add_gtest(test_indexed_render
        SOURCES tests/gpu/state/test_indexed_render.cpp
        INCLUDES tests frontends
        LIBRARIES prosper_core Vulkan::Vulkan)
      add_test(NAME indexed_render_no_index_copy_batch COMMAND test_indexed_render)
      set_tests_properties(indexed_render_no_index_copy_batch PROPERTIES
        ENVIRONMENT "PROSPER_NO_INDEX_COPY_BATCH=1")
      add_test(NAME indexed_render_no_index_arena COMMAND test_indexed_render)
      set_tests_properties(indexed_render_no_index_arena PROPERTIES
        ENVIRONMENT "PROSPER_NO_INDEX_ARENA=1")

      add_executable(test_descriptor_array_render tests/gpu/state/test_descriptor_array_render.cpp)
      # Includes a shared fixture, which resolves from the `tests` root.
      target_include_directories(test_descriptor_array_render PRIVATE tests)
      # Reaches `shared/...` through render_runner.h, which resolves from `frontends`.
      target_include_directories(test_descriptor_array_render PRIVATE frontends)
      target_link_libraries(test_descriptor_array_render PRIVATE prosper_core Vulkan::Vulkan)
      add_test(NAME descriptor_array_render COMMAND test_descriptor_array_render)

      # Storage images: a recompiled compute kernel does MIMG image_load->image_store (no sampler) to
      # copy a 1D storage image bit-exact (the game's blit/copy compute-shader class).
      # Includes a shared fixture, which resolves from the `tests` root.
      prosper_add_gtest(test_storage_image_copy
        SOURCES tests/gpu/test_storage_image_copy.cpp
        INCLUDES tests
        LIBRARIES prosper_core Vulkan::Vulkan)

      # Full textured-draw path: VS exports interpolated UVs, PS samples a texture at them (+screenshot).
      # Includes a shared fixture, which resolves from the `tests` root.
      # Reaches `shared/...` through render_runner.h, which resolves from `frontends`.
      prosper_add_gtest(test_textured_interp_render
        SOURCES tests/gpu/test_textured_interp_render.cpp
        INCLUDES tests frontends
        LIBRARIES prosper_core Vulkan::Vulkan)

      # The sampled-depth bridge's miss classifier must name the reason belonging to the entry it
      # rejected, not the first failing reason across every sibling at that address.
      # Includes a shared fixture, which resolves from the `tests` root.
      # Reaches `shared/...` through render_runner.h, which resolves from `frontends`.
      prosper_add_gtest(test_dsbridge_miss_classifier
        SOURCES tests/render/test_dsbridge_miss_classifier.cpp
        INCLUDES tests frontends
        LIBRARIES prosper_core Vulkan::Vulkan)

      # IMAGE_SAMPLE_C_LZ 2D (shadow-map compare, #1271): manual-dref lowering, end-to-end.
      # Includes a shared fixture, which resolves from the `tests` root.
      # Reaches `shared/...` through render_runner.h, which resolves from `frontends`.
      prosper_add_gtest(test_shadow_compare_render
        SOURCES tests/gpu/test_shadow_compare_render.cpp
        INCLUDES tests frontends
        LIBRARIES prosper_core Vulkan::Vulkan)

      # Generated mip chains bounded by the T#'s declared levels (#1272), end-to-end.
      # Includes a shared fixture, which resolves from the `tests` root.
      # Reaches `shared/...` through render_runner.h, which resolves from `frontends`.
      prosper_add_gtest(test_texture_mip_render
        SOURCES tests/gpu/test_texture_mip_render.cpp
        INCLUDES tests frontends
        LIBRARIES prosper_core Vulkan::Vulkan)

      # Pipeline realization: build a VkPipeline whose fixed-function state comes from
      # resolve_pipeline_state(RenderState) and prove the resolved state drives rendering.
      # Includes a shared fixture, which resolves from the `tests` root.
      # Reaches `shared/...` through render_runner.h, which resolves from `frontends`.
      prosper_add_gtest(test_pipeline_render
        SOURCES tests/gpu/test_pipeline_render.cpp
        INCLUDES tests frontends
        LIBRARIES prosper_core Vulkan::Vulkan)

      add_executable(test_pipeline_cache_file frontends/shared/tests/test_pipeline_cache_file.cpp)
      target_include_directories(test_pipeline_cache_file PRIVATE frontends)
      target_link_libraries(test_pipeline_cache_file PRIVATE Vulkan::Vulkan Threads::Threads)
      if(Python3_Interpreter_FOUND)
        add_test(NAME pipeline_cache_file COMMAND ${Python3_EXECUTABLE}
          ${CMAKE_CURRENT_SOURCE_DIR}/frontends/shared/tests/run_pipeline_cache_file.py
          $<TARGET_FILE:test_pipeline_cache_file>)
      endif()

      add_executable(test_graphics_cache_persistence tests/gpu/test_graphics_cache_persistence.cpp)
      target_include_directories(test_graphics_cache_persistence PRIVATE tests frontends)
      target_link_libraries(test_graphics_cache_persistence PRIVATE prosper_core Vulkan::Vulkan)
      if(Python3_Interpreter_FOUND)
        add_test(NAME graphics_cache_persistence COMMAND ${Python3_EXECUTABLE}
          ${CMAKE_CURRENT_SOURCE_DIR}/tests/gpu/test_graphics_cache_persistence.py
          $<TARGET_FILE:test_graphics_cache_persistence>)
      endif()

      # Full back-half spine: a GpuState -> recompile shaders from their PGM addresses -> resolve
      # pipeline state -> rendered frame (the path a real submitted Dcb will drive).
      # Includes a shared fixture, which resolves from the `tests` root.
      # Reaches `shared/...` through render_runner.h, which resolves from `frontends`.
      prosper_add_gtest(test_gpustate_render
        SOURCES tests/gpu/test_gpustate_render.cpp
        INCLUDES tests frontends
        LIBRARIES prosper_core Vulkan::Vulkan)

      # #3225: the backend's queue wrappers really consult the shutdown gate. The pure gate test
      # (gpu_submit_gate) asserts the gate's logic; this asserts the wiring, which needs
      # render_runner.h. Its positive arm uses a real queue when one exists and is skipped with a
      # printed note when it does not — the refusal arm needs no device at all.
      # Includes a shared fixture, which resolves from the `tests` root.
      # Reaches `shared/...` through render_runner.h, which resolves from `frontends`.
      prosper_add_gtest(test_submit_gate_wiring
        SOURCES tests/gpu/test_submit_gate_wiring.cpp
        INCLUDES tests frontends
        LIBRARIES prosper_core Vulkan::Vulkan)

      # GPU-executor core (Stage A): GpuState -> execute_gpustate() -> render -> present round-trip.
      # Includes a shared fixture, which resolves from the `tests` root.
      # Reaches `shared/...` through render_runner.h, which resolves from `frontends`.
      add_executable(test_gpu_execute tests/gpu/execute/test_gpu_execute.cpp)
      # Includes a shared fixture, which resolves from the `tests` root.
      target_include_directories(test_gpu_execute PRIVATE tests)
      # Reaches `shared/...` through render_runner.h, which resolves from `frontends`.
      target_include_directories(test_gpu_execute PRIVATE frontends)
      target_link_libraries(test_gpu_execute PRIVATE prosper_core Vulkan::Vulkan)
      add_test(NAME gpu_execute COMMAND test_gpu_execute)

      # Payoff demo: recompile the GAME'S OWN embedded shaders and render them on real Vulkan.
      # Needs both Vulkan and the (gitignored) game dump, so it's gated on HAVE_GAME_DUMP.
      if(HAVE_GAME_DUMP)
        add_executable(test_real_shader_render tests/self/test_real_shader_render.cpp)
        # Includes a shared fixture, which resolves from the `tests` root.
        target_include_directories(test_real_shader_render PRIVATE tests)
        # Reaches `shared/...` through render_runner.h, which resolves from `frontends`.
        target_include_directories(test_real_shader_render PRIVATE frontends)
        target_link_libraries(test_real_shader_render PRIVATE prosper_core Vulkan::Vulkan)
        # Messenger-specific: it recompiles and renders the RDNA2 blobs embedded in THIS title's
        # eboot, so a dump whose eboot carries none (or none that fully translate) fails (#1573).
        prosper_add_title_gated_test(real_shader_render PPSA24651
                 test_real_shader_render "${GAME_DUMP}/eboot.bin")
      endif()
      message(STATUS "Vulkan found (${Vulkan_LIBRARY}) — offscreen graphics harness enabled")
    else()
      message(STATUS "Vulkan not found — skipping offscreen graphics harness")
    endif()
  endif()
endif()

if(TARGET prosper_core AND TARGET Vulkan::Vulkan)
  # vkval_sync_probe (#3248): the positive control for `vk_validation_scan.py --sync`. It
  # commits one deliberate write-after-write so the scanner can prove SYNCHRONIZATION validation
  # is armed before reporting a sync verdict -- loading the layer proves only that CORE checks
  # run, and a sync setting the layer version does not recognise produces exactly the output of
  # a clean sync run. Deliberately NOT registered as a ctest: it exists to be invalid, and the
  # gate it feeds fails on any hazard the ledger does not carry.
  add_executable(vkval_sync_probe tools/vkval/sync_probe.cpp)
  target_link_libraries(vkval_sync_probe PRIVATE Vulkan::Vulkan)

  # Interpolation: a recompiled VS exports EXP PARAM0, a recompiled PS reads it via v_interp.
  # Includes a shared fixture, which resolves from the `tests` root.
  # Reaches `shared/...` through render_runner.h, which resolves from `frontends`.
  prosper_add_gtest(test_interp_render
    SOURCES tests/gpu/recompiler/test_interp_render.cpp
    INCLUDES tests frontends
    LIBRARIES prosper_core Vulkan::Vulkan)


  # RectList fourth-corner synthesis: the geometry stage must complete the rectangle whichever of
  # the three supplied vertices is the shared corner (#3507).
  # Includes a shared fixture, which resolves from the `tests` root.
  # Reaches `shared/...` through render_runner.h, which resolves from `frontends`.
  prosper_add_gtest(test_rect_synthesis_render
    SOURCES tests/gpu/recompiler/test_rect_synthesis_render.cpp
    INCLUDES tests frontends
    LIBRARIES prosper_core Vulkan::Vulkan)

  # CPU-only controls drive the shipping persistent-color lookup/readback refusal branches.
  # Vulkan dev files satisfy the backend header; no usable device or GPU execution is required.
  add_executable(test_persistent_readback_refusal tests/render/test_persistent_readback_refusal.cpp)
  target_include_directories(test_persistent_readback_refusal PRIVATE tests frontends)
  target_link_libraries(test_persistent_readback_refusal PRIVATE prosper_core Vulkan::Vulkan)
  add_test(NAME persistent_readback_refusal COMMAND test_persistent_readback_refusal --armed)
  add_test(NAME persistent_readback_refusal_disabled COMMAND ${CMAKE_COMMAND} -E env
    --unset=PROSPER_RTT_READBACK_TRACE_TARGETS $<TARGET_FILE:test_persistent_readback_refusal> --unarmed)
  set_tests_properties(persistent_readback_refusal PROPERTIES TIMEOUT 15
    ENVIRONMENT "PROSPER_RTT_READBACK_TRACE_TARGETS=0x38910000")
  set_tests_properties(persistent_readback_refusal_disabled PROPERTIES TIMEOUT 15)
  set(readback_refusal_selector_malformed "0x38910000,bad")
  set(readback_refusal_selector_empty "")
  set(readback_refusal_selector_zero "0x38910000,0x0")
  set(readback_refusal_selector_excess "0x38910000,0x1,0x2,0x3,0x4,0x5,0x6,0x7,0x8")
  foreach(readback_refusal_selector_mode IN ITEMS empty malformed zero excess)
    add_test(NAME persistent_readback_refusal_${readback_refusal_selector_mode}
      COMMAND test_persistent_readback_refusal --unarmed)
    set_tests_properties(persistent_readback_refusal_${readback_refusal_selector_mode}
      PROPERTIES TIMEOUT 15
      ENVIRONMENT "PROSPER_RTT_READBACK_TRACE_TARGETS=${readback_refusal_selector_${readback_refusal_selector_mode}}")
  endforeach()

  # A compute full-overwrite destination at the colour-target COUNT bound evicts under the graphics
  # admission predicate instead of refusing (#3873).
  # BackendSubmissionBatch's timestamp envelope reuses one query pool across batches (#3948).
  add_executable(test_backend_timestamp_pool tests/gpu/state/test_backend_timestamp_pool.cpp)
  target_include_directories(test_backend_timestamp_pool PRIVATE tests frontends)
  target_link_libraries(test_backend_timestamp_pool PRIVATE prosper_core Vulkan::Vulkan)
  add_test(NAME backend_timestamp_pool COMMAND test_backend_timestamp_pool)
  set_tests_properties(backend_timestamp_pool PROPERTIES SKIP_RETURN_CODE 77)
  add_executable(test_compute_overwrite_eviction tests/gpu/state/test_compute_overwrite_eviction.cpp)
  target_include_directories(test_compute_overwrite_eviction PRIVATE tests frontends)
  target_link_libraries(test_compute_overwrite_eviction PRIVATE prosper_core Vulkan::Vulkan)
  add_test(NAME compute_overwrite_eviction COMMAND ${CMAKE_COMMAND} -E env
    PROSPER_BACKEND_TARGET_CACHE_COUNT=4 --unset=PROSPER_NO_COMPUTE_RTT_DEST_EVICT
    --unset=PROSPER_NO_BACKEND_PERSISTENT_COLOR_TARGETS
    $<TARGET_FILE:test_compute_overwrite_eviction>)
  set_tests_properties(compute_overwrite_eviction PROPERTIES SKIP_RETURN_CODE 77)
  add_executable(test_render_buffer_residency tests/gpu/state/test_render_buffer_residency.cpp)
  target_include_directories(test_render_buffer_residency PRIVATE tests frontends)
  target_link_libraries(test_render_buffer_residency PRIVATE prosper_core Vulkan::Vulkan)
  foreach(lookup_control IN ITEMS on off)
    set(lookup_environment --unset=PROSPER_NO_BACKEND_RESOURCE_SHARE
      --unset=PROSPER_NO_BACKEND_BUFFER_POOL --unset=PROSPER_BUFVERIFY
      --unset=PROSPER_NO_BUFFER_LOOKUP_ARENA PROSPER_BACKEND_BUFFER_RESIDENCY_MB=0)
    if(lookup_control STREQUAL "off")
      list(APPEND lookup_environment PROSPER_NO_BUFFER_LOOKUP_ARENA=1)
    endif()
    add_test(NAME render_buffer_lookup_${lookup_control} COMMAND ${CMAKE_COMMAND} -E env
      ${lookup_environment} $<TARGET_FILE:test_render_buffer_residency>
      --lookup-arena ${lookup_control})
  endforeach()
  foreach(range_control IN ITEMS on off arena_off verify)
    set(range_environment PROSPER_BACKEND_BUFFER_RESIDENCY_MB=0
      PROSPER_BACKEND_BUFFER_RANGE_RESIDENCY_MB=0
      --unset=PROSPER_NO_BACKEND_RESOURCE_SHARE --unset=PROSPER_NO_BACKEND_BUFFER_POOL
      --unset=PROSPER_NO_BACKEND_BUFFER_ARENA --unset=PROSPER_BUFVERIFY
      --unset=PROSPER_NO_BACKEND_BUFFER_RANGE_SHARE
      --unset=PROSPER_NO_BACKEND_BUFFER_COPY_BATCH)
    set(range_expected off)
    if(range_control STREQUAL "on")
      set(range_expected on)
    elseif(range_control STREQUAL "off")
      list(APPEND range_environment PROSPER_NO_BACKEND_BUFFER_RANGE_SHARE=1)
    elseif(range_control STREQUAL "arena_off")
      list(APPEND range_environment PROSPER_NO_BACKEND_BUFFER_ARENA=1)
    elseif(range_control STREQUAL "verify")
      list(APPEND range_environment PROSPER_BUFVERIFY=1)
    endif()
    add_test(NAME render_buffer_range_${range_control} COMMAND ${CMAKE_COMMAND} -E env
      ${range_environment} $<TARGET_FILE:test_render_buffer_residency>
      --range-sharing ${range_expected})
  endforeach()
  add_test(NAME render_buffer_copy_batch_helper COMMAND ${CMAKE_COMMAND} -E env
    PROSPER_RENDER_COPY_THREADS=4
    $<TARGET_FILE:test_render_buffer_residency> --copy-batch-helper)
  foreach(copy_batch_control IN ITEMS on off)
    set(copy_batch_environment PROSPER_BACKEND_BUFFER_RESIDENCY_MB=0
      PROSPER_BACKEND_BUFFER_RANGE_RESIDENCY_MB=0
      PROSPER_RENDER_COPY_THREADS=4
      --unset=PROSPER_NO_BACKEND_RESOURCE_SHARE --unset=PROSPER_NO_BACKEND_BUFFER_POOL
      --unset=PROSPER_NO_BACKEND_BUFFER_ARENA --unset=PROSPER_BUFVERIFY
      --unset=PROSPER_NO_BACKEND_BUFFER_RANGE_SHARE
      --unset=PROSPER_NO_BACKEND_BUFFER_COPY_BATCH)
    if(copy_batch_control STREQUAL "off")
      list(APPEND copy_batch_environment PROSPER_NO_BACKEND_BUFFER_COPY_BATCH=1)
    endif()
    add_test(NAME render_buffer_copy_batch_${copy_batch_control} COMMAND ${CMAKE_COMMAND} -E env
      ${copy_batch_environment} $<TARGET_FILE:test_render_buffer_residency>
      --range-copy-batch ${copy_batch_control})
  endforeach()
  foreach(range_residency_control IN ITEMS on off)
    set(range_residency_environment PROSPER_BACKEND_BUFFER_RESIDENCY_MB=0
      PROSPER_BACKEND_BUFFER_RANGE_RESIDENCY_MB=64
      --unset=PROSPER_BACKEND_BUFFER_RANGE_MIN_BYTES
      --unset=PROSPER_BACKEND_BUFFER_RANGE_RECLAIM
      --unset=PROSPER_NO_BACKEND_BUFFER_RANGE_RECLAIM
      --unset=PROSPER_TEST_RENDER_BUFFER_RECLAIM_FAIL_AFTER_EVICT
      --unset=PROSPER_NO_BACKEND_BUFFER_RESIDENCY
      --unset=PROSPER_NO_BACKEND_BUFFER_RANGE_RESIDENCY
      --unset=PROSPER_NO_BACKEND_RESOURCE_SHARE
      --unset=PROSPER_NO_BACKEND_BUFFER_POOL --unset=PROSPER_BUFVERIFY
      --unset=PROSPER_NO_BACKEND_BUFFER_ARENA
      --unset=PROSPER_NO_BACKEND_BUFFER_RANGE_SHARE)
    if(range_residency_control STREQUAL "off")
      list(APPEND range_residency_environment PROSPER_NO_BACKEND_BUFFER_RANGE_RESIDENCY=1)
    endif()
    add_test(NAME render_buffer_range_residency_${range_residency_control}
      COMMAND ${CMAKE_COMMAND} -E env ${range_residency_environment}
      $<TARGET_FILE:test_render_buffer_residency> --range-residency ${range_residency_control})
  endforeach()
  foreach(range_reclaim_control IN ITEMS on off fail)
    set(range_reclaim_environment PROSPER_BACKEND_BUFFER_RESIDENCY_MB=0
      PROSPER_BACKEND_BUFFER_RANGE_RESIDENCY_MB=64
      PROSPER_BACKEND_BUFFER_RESIDENCY_OWNERS=1
      --unset=PROSPER_BACKEND_BUFFER_RANGE_MIN_BYTES
      --unset=PROSPER_NO_BACKEND_BUFFER_RESIDENCY
      --unset=PROSPER_NO_BACKEND_BUFFER_RANGE_RESIDENCY
      --unset=PROSPER_NO_BACKEND_RESOURCE_SHARE
      --unset=PROSPER_NO_BACKEND_BUFFER_POOL --unset=PROSPER_BUFVERIFY
      --unset=PROSPER_NO_BACKEND_BUFFER_ARENA
      --unset=PROSPER_NO_BACKEND_BUFFER_RANGE_SHARE
      --unset=PROSPER_BACKEND_BUFFER_RANGE_RECLAIM
      --unset=PROSPER_NO_BACKEND_BUFFER_RANGE_RECLAIM
      --unset=PROSPER_TEST_RENDER_BUFFER_RECLAIM_FAIL_AFTER_EVICT)
    if(NOT range_reclaim_control STREQUAL "off")
      list(APPEND range_reclaim_environment PROSPER_BACKEND_BUFFER_RANGE_RECLAIM=1)
    endif()
    if(range_reclaim_control STREQUAL "fail")
      list(APPEND range_reclaim_environment
        PROSPER_TEST_RENDER_BUFFER_RECLAIM_FAIL_AFTER_EVICT=1)
    endif()
    add_test(NAME render_buffer_range_reclaim_${range_reclaim_control}
      COMMAND ${CMAKE_COMMAND} -E env ${range_reclaim_environment}
      $<TARGET_FILE:test_render_buffer_residency> --range-reclaim ${range_reclaim_control})
  endforeach()
  add_test(NAME render_buffer_range_minimum
    COMMAND ${CMAKE_COMMAND} -E env
      PROSPER_BACKEND_BUFFER_RESIDENCY_MB=0
      PROSPER_BACKEND_BUFFER_RANGE_RESIDENCY_MB=64
      PROSPER_BACKEND_BUFFER_RANGE_MIN_BYTES=10000
      PROSPER_BACKEND_BUFFER_RESIDENCY_OWNERS=1
      --unset=PROSPER_BACKEND_BUFFER_RANGE_RECLAIM
      --unset=PROSPER_NO_BACKEND_BUFFER_RANGE_RECLAIM
      --unset=PROSPER_TEST_RENDER_BUFFER_RECLAIM_FAIL_AFTER_EVICT
      --unset=PROSPER_NO_BACKEND_BUFFER_RESIDENCY
      --unset=PROSPER_NO_BACKEND_BUFFER_RANGE_RESIDENCY
      --unset=PROSPER_NO_BACKEND_RESOURCE_SHARE
      --unset=PROSPER_NO_BACKEND_BUFFER_POOL --unset=PROSPER_BUFVERIFY
      --unset=PROSPER_NO_BACKEND_BUFFER_ARENA
      --unset=PROSPER_NO_BACKEND_BUFFER_RANGE_SHARE
      $<TARGET_FILE:test_render_buffer_residency> --range-minimum)
  add_test(NAME render_buffer_range_residency_global_disabled
    COMMAND ${CMAKE_COMMAND} -E env
      PROSPER_BACKEND_BUFFER_RESIDENCY_MB=0
      PROSPER_BACKEND_BUFFER_RANGE_RESIDENCY_MB=64
      PROSPER_NO_BACKEND_BUFFER_RESIDENCY=1
      --unset=PROSPER_NO_BACKEND_BUFFER_RANGE_RESIDENCY
      --unset=PROSPER_NO_BACKEND_RESOURCE_SHARE
      --unset=PROSPER_NO_BACKEND_BUFFER_POOL --unset=PROSPER_BUFVERIFY
      --unset=PROSPER_NO_BACKEND_BUFFER_ARENA
      --unset=PROSPER_NO_BACKEND_BUFFER_RANGE_SHARE
      $<TARGET_FILE:test_render_buffer_residency> --range-residency off)
  add_test(NAME render_buffer_range_residency_owner_exhausted
    COMMAND ${CMAKE_COMMAND} -E env
      PROSPER_BACKEND_BUFFER_RESIDENCY_MB=0
      PROSPER_BACKEND_BUFFER_RANGE_RESIDENCY_MB=64
      PROSPER_BACKEND_BUFFER_RESIDENCY_OWNERS=0
      --unset=PROSPER_NO_BACKEND_BUFFER_RESIDENCY
      --unset=PROSPER_NO_BACKEND_BUFFER_RANGE_RESIDENCY
      --unset=PROSPER_NO_BACKEND_RESOURCE_SHARE
      --unset=PROSPER_NO_BACKEND_BUFFER_POOL --unset=PROSPER_BUFVERIFY
      --unset=PROSPER_NO_BACKEND_BUFFER_ARENA
      --unset=PROSPER_NO_BACKEND_BUFFER_RANGE_SHARE
      $<TARGET_FILE:test_render_buffer_residency> --range-residency off)
  add_test(NAME render_buffer_residency COMMAND ${CMAKE_COMMAND} -E env
    --unset=PROSPER_NO_BACKEND_RESOURCE_SHARE
    --unset=PROSPER_NO_BACKEND_BUFFER_POOL --unset=PROSPER_BUFVERIFY
    PROSPER_BACKEND_BUFFER_RANGE_RESIDENCY_MB=0
    PROSPER_BACKEND_BUFFER_RESIDENCY_MB=256
    PROSPER_BACKEND_BUFFER_RESIDENCY_OWNERS=4096
    $<TARGET_FILE:test_render_buffer_residency>)
  foreach(owner_limit IN ITEMS 0 256)
    add_test(NAME render_buffer_owner_limit_${owner_limit} COMMAND ${CMAKE_COMMAND} -E env
      PROSPER_BACKEND_BUFFER_RESIDENCY_OWNERS=${owner_limit}
      $<TARGET_FILE:test_render_buffer_residency> --configured-owner-limit ${owner_limit})
  endforeach()
  add_test(NAME render_buffer_residency_default COMMAND ${CMAKE_COMMAND} -E env
    --unset=PROSPER_BACKEND_BUFFER_RESIDENCY_MB
    --unset=PROSPER_BACKEND_BUFFER_RANGE_RESIDENCY_MB
    --unset=PROSPER_BACKEND_BUFFER_RANGE_MIN_BYTES
    --unset=PROSPER_BACKEND_BUFFER_RANGE_RECLAIM
    --unset=PROSPER_NO_BACKEND_BUFFER_RANGE_RECLAIM
    --unset=PROSPER_NO_BACKEND_BUFFER_RANGE_RESIDENCY
    --unset=PROSPER_NO_BACKEND_BUFFER_RESIDENCY
    --unset=PROSPER_NO_BACKEND_BUFFER_WRITE_WATCH
    --unset=PROSPER_BACKEND_BUFFER_RESIDENCY_OWNERS
    --unset=PROSPER_NO_BACKEND_RESOURCE_SHARE
    --unset=PROSPER_NO_BACKEND_BUFFER_POOL --unset=PROSPER_BUFVERIFY
    $<TARGET_FILE:test_render_buffer_residency> --default-disabled)
  prosper_add_gtest(test_gpu_detile
    SOURCES tests/gpu/execute/test_gpu_detile.cpp
    INCLUDES tests
    LIBRARIES prosper_core Vulkan::Vulkan)
  # The bounded VkCommandPool free list (#3407). Registered TWICE: the second run sets
  # PROSPER_NO_COMMAND_POOL_REUSE, which is the negative control and has to be a separate process
  # because the switch is read once into a function-local static.
  prosper_add_gtest(test_command_pool_reuse
    SOURCES tests/gpu/execute/test_command_pool_reuse.cpp
    INCLUDES tests frontends
    LIBRARIES prosper_core Vulkan::Vulkan)
  add_test(NAME command_pool_reuse_disabled COMMAND test_command_pool_reuse)
  set_tests_properties(command_pool_reuse_disabled PROPERTIES
                       ENVIRONMENT "PROSPER_NO_COMMAND_POOL_REUSE=1")

  add_executable(test_mapped_staging tests/gpu/execute/test_mapped_staging.cpp)
  target_include_directories(test_mapped_staging PRIVATE tests frontends)
  target_link_libraries(test_mapped_staging PRIVATE prosper_core Vulkan::Vulkan)
  add_test(NAME mapped_staging COMMAND test_mapped_staging)
  set_tests_properties(mapped_staging PROPERTIES ENVIRONMENT "PROSPER_MAPPED_STAGING_MB=2;PROSPER_MAPPED_STAGING_LOG=1")
endif()

if(TARGET prosper_live_renderer AND TARGET Vulkan::Vulkan)
  # #4272: the owned-wave transaction controls use portable HLE and the shared renderer.
  # Register once after both platform frontend blocks, not only on UNIX.
  prosper_add_gtest(test_owned_graphics_wave_live
    SOURCES tests/shared/live/test_owned_graphics_wave_live.cpp
    INCLUDES frontends src tests
    LIBRARIES prosper_core prosper_live_renderer Vulkan::Vulkan)
  # #4106: same shipping DrawItem caller, with an explicit device-free ownership arm.
  add_executable(test_raster_quad_collection tests/shared/live/test_raster_quad_collection.cpp)
  target_include_directories(test_raster_quad_collection PRIVATE src tests frontends)
  target_link_libraries(test_raster_quad_collection PRIVATE prosper_core prosper_live_renderer Vulkan::Vulkan)
  add_test(NAME raster_quad_realization COMMAND test_raster_quad_collection --cpu-only)
  add_test(NAME raster_quad_realization_analysis_unavailable
    COMMAND test_raster_quad_collection --cpu-only --analysis-unavailable)
  add_test(NAME raster_quad_realization_scissor_override
    COMMAND test_raster_quad_collection --cpu-only --scissor-override)
  add_test(NAME raster_quad_collection COMMAND test_raster_quad_collection)
  add_test(NAME raster_quad_collection_scissor_override COMMAND test_raster_quad_collection --scissor-override)
  set(_raster_quad_clean_env "PROSPER_NO_SHADER_DECODE_REUSE=unset:;PROSPER_RENDER_REFVS=unset:;PROSPER_FS_SPV=unset:;PROSPER_RENDER_TESTPS=unset:;PROSPER_RENDER_NOPS=unset:;PROSPER_DESCRIPTOR_VALIDATE=unset:;PROSPER_NO_CULL=unset:;PROSPER_FLIP_FRONT_FACE=unset:;PROSPER_NO_DEPTH_BIAS=unset:;PROSPER_SKIP_DRAW=unset:;PROSPER_RENDER_FIRST=unset:;PROSPER_RENDER_LAST=unset:")
  set_tests_properties(raster_quad_realization raster_quad_collection PROPERTIES
    ENVIRONMENT "PROSPER_FRAGMENT_QUAD_COLLECT=1"
    ENVIRONMENT_MODIFICATION "${_raster_quad_clean_env};PROSPER_NO_FRAGMENT_ANALYSIS_REUSE=unset:;PROSPER_IGNORE_EMPTY_SCISSOR=unset:"
    TIMEOUT 90)
  set_tests_properties(raster_quad_collection raster_quad_collection_scissor_override PROPERTIES SKIP_RETURN_CODE 77)
  set_tests_properties(raster_quad_realization_analysis_unavailable PROPERTIES
    ENVIRONMENT "PROSPER_FRAGMENT_QUAD_COLLECT=1;PROSPER_NO_FRAGMENT_ANALYSIS_REUSE=1"
    ENVIRONMENT_MODIFICATION "${_raster_quad_clean_env};PROSPER_IGNORE_EMPTY_SCISSOR=unset:" TIMEOUT 90)
  set_tests_properties(raster_quad_realization_scissor_override raster_quad_collection_scissor_override PROPERTIES
    ENVIRONMENT "PROSPER_FRAGMENT_QUAD_COLLECT=1;PROSPER_IGNORE_EMPTY_SCISSOR=0"
    ENVIRONMENT_MODIFICATION "${_raster_quad_clean_env};PROSPER_NO_FRAGMENT_ANALYSIS_REUSE=unset:" TIMEOUT 90)
  unset(_raster_quad_clean_env)
  add_executable(test_render_buffer_capture tests/shared/live/test_render_buffer_capture.cpp)
  target_include_directories(test_render_buffer_capture PRIVATE frontends)
  target_link_libraries(test_render_buffer_capture PRIVATE prosper_live_renderer
                                                         prosper_performance_capture Vulkan::Vulkan)
  add_test(NAME render_buffer_capture COMMAND test_render_buffer_capture)
  add_test(NAME render_buffer_range_capture COMMAND test_render_buffer_capture --range-sharing)
  set_tests_properties(render_buffer_range_capture PROPERTIES
    ENVIRONMENT "PROSPER_NO_LIVE_PERSISTENT_COLOR_TARGETS=1;PROSPER_BACKEND_BUFFER_RESIDENCY_MB=0"
    TIMEOUT 120)
  add_test(NAME render_texture_snapshot_capture COMMAND test_render_buffer_capture --source-snapshot)
  add_test(NAME render_texture_direct_validation COMMAND test_render_buffer_capture --direct-validation)
  add_test(NAME render_texture_direct_validation_scratch COMMAND test_render_buffer_capture --direct-validation)
  set_tests_properties(render_texture_direct_validation PROPERTIES
    ENVIRONMENT "PROSPER_NO_LIVE_PERSISTENT_COLOR_TARGETS=1;PROSPER_BACKEND_BUFFER_RESIDENCY_MB=256;PROSPER_BACKEND_BUFFER_RESIDENCY_OWNERS=4096;PROSPER_TEXTURE_WRITE_WATCH_MIN_KB=1024;PROSPER_NO_SUBMIT_TEXTURE_VALIDATION_REUSE=1" TIMEOUT 120)
  set_tests_properties(render_texture_direct_validation_scratch PROPERTIES
    ENVIRONMENT "PROSPER_NO_LIVE_PERSISTENT_COLOR_TARGETS=1;PROSPER_BACKEND_BUFFER_RESIDENCY_MB=256;PROSPER_BACKEND_BUFFER_RESIDENCY_OWNERS=4096;PROSPER_TEXTURE_WRITE_WATCH_MIN_KB=1024;PROSPER_NO_SUBMIT_TEXTURE_VALIDATION_REUSE=1;PROSPER_TEXTURE_VALIDATION_SCRATCH_COPY=1" TIMEOUT 120)
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    add_test(NAME render_texture_direct_validation_watch COMMAND test_render_buffer_capture --direct-validation-watch)
    set_tests_properties(render_texture_direct_validation_watch PROPERTIES
      ENVIRONMENT "PROSPER_NO_LIVE_PERSISTENT_COLOR_TARGETS=1;PROSPER_BACKEND_BUFFER_RESIDENCY_MB=0;PROSPER_NO_SUBMIT_TEXTURE_VALIDATION_REUSE=1" TIMEOUT 120)
    add_test(NAME render_small_texture_watch COMMAND test_render_buffer_capture --small-watch)
    add_test(NAME render_small_texture_watch_zero_hits COMMAND test_render_buffer_capture --small-watch)
    add_test(NAME render_small_texture_watch_unsupported COMMAND test_render_buffer_capture --small-watch-unsupported)
    add_test(NAME render_small_texture_watch_control COMMAND test_render_buffer_capture --small-watch-control)
    set_tests_properties(render_small_texture_watch render_small_texture_watch_unsupported PROPERTIES
      ENVIRONMENT "PROSPER_NO_LIVE_PERSISTENT_COLOR_TARGETS=1;PROSPER_BACKEND_BUFFER_RESIDENCY_MB=0" TIMEOUT 120)
    set_tests_properties(render_small_texture_watch_zero_hits PROPERTIES
      ENVIRONMENT "PROSPER_NO_LIVE_PERSISTENT_COLOR_TARGETS=1;PROSPER_BACKEND_BUFFER_RESIDENCY_MB=0;PROSPER_TEXTURE_WRITE_WATCH_PROMOTE_HITS=0" TIMEOUT 120)
    set_tests_properties(render_small_texture_watch_control PROPERTIES
      ENVIRONMENT "PROSPER_NO_LIVE_PERSISTENT_COLOR_TARGETS=1;PROSPER_BACKEND_BUFFER_RESIDENCY_MB=0;PROSPER_NO_SMALL_TEXTURE_WRITE_WATCH=1" TIMEOUT 120)
  endif()
  if(Python3_Interpreter_FOUND)
    add_test(NAME render_texture_validation_census
      COMMAND "${Python3_EXECUTABLE}"
        "${CMAKE_SOURCE_DIR}/tests/shared/live/test_texture_validation_census.py"
        $<TARGET_FILE:test_render_buffer_capture>)
    set_tests_properties(render_texture_validation_census PROPERTIES TIMEOUT 150)
  endif()
  add_test(NAME render_texture_snapshot_capture_copy COMMAND test_render_buffer_capture --source-snapshot)
  set_tests_properties(render_texture_snapshot_capture PROPERTIES
    ENVIRONMENT "PROSPER_NO_LIVE_PERSISTENT_COLOR_TARGETS=1;PROSPER_BACKEND_BUFFER_RESIDENCY_MB=256;PROSPER_BACKEND_BUFFER_RESIDENCY_OWNERS=4096" TIMEOUT 120)
  set_tests_properties(render_texture_snapshot_capture_copy PROPERTIES
    ENVIRONMENT "PROSPER_NO_LIVE_PERSISTENT_COLOR_TARGETS=1;PROSPER_BACKEND_BUFFER_RESIDENCY_MB=256;PROSPER_BACKEND_BUFFER_RESIDENCY_OWNERS=4096;PROSPER_NO_TEXTURE_SOURCE_SNAPSHOT_MOVE=1" TIMEOUT 120)
  set_tests_properties(render_buffer_capture PROPERTIES
    ENVIRONMENT "PROSPER_NO_LIVE_PERSISTENT_COLOR_TARGETS=1;PROSPER_BACKEND_BUFFER_RESIDENCY_MB=256;PROSPER_BACKEND_BUFFER_RESIDENCY_OWNERS=4096" TIMEOUT 120)

  # Execution coverage for the result-compare shader. Its own target because it drives the shader
  # against a plain Vulkan device rather than through the live backend -- a failure should indict the
  # module, not the backend's resource machinery.
  add_executable(test_compute_result_compare tests/gpu/execute/test_compute_result_compare.cpp)
  target_include_directories(test_compute_result_compare PRIVATE tests frontends)
  target_link_libraries(test_compute_result_compare PRIVATE prosper_live_renderer Vulkan::Vulkan)
  add_test(NAME compute_result_compare COMMAND test_compute_result_compare)
  add_test(NAME compute_retile_result_compare COMMAND test_compute_result_compare --retile)
  set_tests_properties(compute_result_compare compute_retile_result_compare PROPERTIES SKIP_RETURN_CODE 77)

  add_executable(test_gpu_retile tests/gpu/execute/test_gpu_retile.cpp)
  target_include_directories(test_gpu_retile PRIVATE tests frontends)
  target_link_libraries(test_gpu_retile PRIVATE prosper_live_renderer Vulkan::Vulkan)
  add_test(NAME gpu_retile COMMAND test_gpu_retile)
  add_test(NAME gpu_retile_clean COMMAND test_gpu_retile --clean)
  foreach(mode direct direct-clean direct-host direct-readback direct-alloc-oom direct-map-oom
               direct-alloc-loss direct-map-loss direct-compare-fallback direct-compare-loss direct-compare-transfer)
    add_test(NAME gpu_retile_${mode} COMMAND ${CMAKE_COMMAND} -E env
      --unset=PROSPER_NO_DIRECT_IMAGE_RETILE PROSPER_COMPUTE_STORAGE_IMAGE_CACHE_MIN_KB=1
      $<TARGET_FILE:test_gpu_retile> --${mode})
    set_tests_properties(gpu_retile_${mode} PROPERTIES TIMEOUT 60)
  endforeach()
  add_test(NAME gpu_retile_compare_control COMMAND ${CMAKE_COMMAND} -E env
    --unset=PROSPER_NO_DIRECT_IMAGE_RETILE PROSPER_COMPUTE_STORAGE_IMAGE_CACHE_MIN_KB=1
    PROSPER_NO_FUSED_RETILE_COMPARE=1 $<TARGET_FILE:test_gpu_retile> --direct)
  set_tests_properties(gpu_retile_compare_control PROPERTIES TIMEOUT 60)
  add_test(NAME gpu_retile_direct_disabled COMMAND ${CMAKE_COMMAND} -E env
    PROSPER_NO_DIRECT_IMAGE_RETILE=1 PROSPER_COMPUTE_STORAGE_IMAGE_CACHE_MIN_KB=1
    $<TARGET_FILE:test_gpu_retile> --direct-disabled)
  set_tests_properties(gpu_retile_direct_disabled PROPERTIES TIMEOUT 60)
  add_test(NAME gpu_retile_direct_writeback_control COMMAND ${CMAKE_COMMAND} -E env
    --unset=PROSPER_NO_DIRECT_IMAGE_RETILE PROSPER_COMPUTE_STORAGE_IMAGE_CACHE_MIN_KB=1
    PROSPER_NO_READONLY_STORAGE_WRITEBACK_SKIP=1 $<TARGET_FILE:test_gpu_retile> --direct)
  set_tests_properties(gpu_retile_direct_writeback_control PROPERTIES TIMEOUT 60)
  add_test(NAME gpu_retile_cpu COMMAND test_gpu_retile --cpu)
  add_test(NAME gpu_retile_single_layer_array_control COMMAND test_gpu_retile)
  set_tests_properties(gpu_retile_single_layer_array_control PROPERTIES
    ENVIRONMENT "PROSPER_NO_NATIVE_SINGLE_LAYER_ARRAY_STORAGE=1")
  add_test(NAME gpu_retile_allocation_loss COMMAND test_gpu_retile --alloc-loss)
  add_test(NAME gpu_retile_mapping_loss COMMAND test_gpu_retile --map-loss)
  add_test(NAME gpu_retile_allocation_fallback COMMAND test_gpu_retile --alloc-oom)
  add_test(NAME gpu_retile_mapping_fallback COMMAND test_gpu_retile --map-oom)
  add_test(NAME gpu_retile_mixed COMMAND test_gpu_retile --mixed)
  add_test(NAME gpu_retile_mixed_writeback_control COMMAND ${CMAKE_COMMAND} -E env
           PROSPER_NO_READONLY_STORAGE_WRITEBACK_SKIP=1 $<TARGET_FILE:test_gpu_retile> --mixed)
  set_tests_properties(gpu_retile_mixed_writeback_control PROPERTIES TIMEOUT 60)
  add_test(NAME gpu_retile_volume COMMAND test_gpu_retile --volume)
  add_test(NAME gpu_retile_volume_cpu COMMAND test_gpu_retile --volume-cpu)
  add_test(NAME gpu_retile_volume_allocation_fallback COMMAND test_gpu_retile --volume-alloc-oom)
  add_test(NAME gpu_retile_volume_mapping_loss COMMAND test_gpu_retile --volume-map-loss)
  foreach(pipes 1 2 4 8 16 32 64)
    add_test(NAME gpu_retile_subword_pipes_${pipes} COMMAND ${CMAKE_COMMAND} -E env
      PROSPER_RX_PIPES=${pipes} $<TARGET_FILE:test_gpu_retile> --subword)
  endforeach()
  add_test(NAME gpu_retile_subword_cpu COMMAND ${CMAKE_COMMAND} -E env
    PROSPER_NO_GPU_RETILE=1 $<TARGET_FILE:test_gpu_retile> --subword-cpu)
  add_test(NAME gpu_retile_subword_allocation_fallback COMMAND test_gpu_retile --subword-alloc-oom)
  add_test(NAME gpu_retile_subword_mapping_loss COMMAND test_gpu_retile --subword-map-loss)
  add_test(NAME gpu_retile_array16 COMMAND test_gpu_retile --array16)
  add_test(NAME gpu_retile_array16_shape_refusal COMMAND test_gpu_retile --array16-shape-refusal)
  add_test(NAME gpu_retile_array16_cpu COMMAND test_gpu_retile --array16-cpu)
  add_test(NAME gpu_retile_array16_allocation_fallback COMMAND test_gpu_retile --array16-alloc-oom)
  add_test(NAME gpu_retile_array16_mapping_fallback COMMAND test_gpu_retile --array16-map-oom)
  add_test(NAME gpu_retile_array16_allocation_loss COMMAND test_gpu_retile --array16-alloc-loss)
  add_test(NAME gpu_retile_array16_mapping_loss COMMAND test_gpu_retile --array16-map-loss)
  set_tests_properties(gpu_retile_array16 gpu_retile_array16_cpu gpu_retile_array16_shape_refusal
      gpu_retile_array16_allocation_fallback gpu_retile_array16_mapping_fallback
      gpu_retile_array16_allocation_loss gpu_retile_array16_mapping_loss PROPERTIES TIMEOUT 120)
  set_tests_properties(gpu_retile_mixed gpu_retile_volume gpu_retile_volume_cpu
                       gpu_retile_volume_allocation_fallback gpu_retile_volume_mapping_loss
                       PROPERTIES TIMEOUT 120)
  set_tests_properties(gpu_retile_cpu gpu_retile_volume_cpu gpu_retile_array16_cpu
                       PROPERTIES ENVIRONMENT "PROSPER_NO_GPU_RETILE=1")
  add_executable(test_gpu_detile_render tests/gpu/execute/test_gpu_detile_render.cpp)
  target_include_directories(test_gpu_detile_render PRIVATE tests frontends)
  target_link_libraries(test_gpu_detile_render PRIVATE prosper_live_renderer Vulkan::Vulkan)
  add_test(NAME gpu_detile_render COMMAND test_gpu_detile_render)
  set_tests_properties(gpu_detile_render PROPERTIES ENVIRONMENT "PROSPER_NO_LIVE_PERSISTENT_COLOR_TARGETS=1")
  add_test(NAME gpu_detile_render_pressure COMMAND test_gpu_detile_render --staging-pressure)
  set_tests_properties(gpu_detile_render_pressure PROPERTIES ENVIRONMENT "PROSPER_NO_LIVE_PERSISTENT_COLOR_TARGETS=1;PROSPER_MAPPED_STAGING_MB=4")
  prosper_add_gtest(test_gpu_detile_storage
    SOURCES tests/gpu/execute/test_gpu_detile_storage.cpp
    INCLUDES tests frontends
    LIBRARIES prosper_live_renderer Vulkan::Vulkan)
  add_test(NAME gpu_detile_storage_cpu COMMAND test_gpu_detile_storage)
  add_test(NAME gpu_detile_storage_unshared COMMAND test_gpu_detile_storage)
  set_tests_properties(gpu_detile_storage_unshared PROPERTIES ENVIRONMENT "PROSPER_NO_BACKEND_TEXTURE_SHARE=1")
  set_tests_properties(gpu_detile_storage_cpu PROPERTIES ENVIRONMENT "PROSPER_NO_GPU_DETILE_2D=1")
  add_test(NAME gpu_detile_render_cpu_2d COMMAND test_gpu_detile_render)
  set_tests_properties(gpu_detile_render_cpu_2d PROPERTIES ENVIRONMENT "PROSPER_NO_LIVE_PERSISTENT_COLOR_TARGETS=1;PROSPER_NO_GPU_DETILE_2D=1")
  prosper_add_gtest(test_broadcast_htile_clear
    SOURCES tests/gpu/execute/test_broadcast_htile_clear.cpp
    INCLUDES tests frontends
    LIBRARIES prosper_live_renderer Vulkan::Vulkan)
endif()

# Normal sampled-render capture on every host with the live renderer. Unix writes PNG through zlib;
# Windows uses its built-in WIC PNG encoder so the native tool needs no extra package.
if(TARGET prosper_live_renderer AND TARGET Vulkan::Vulkan)
  if(WIN32)
    add_executable(screenshot tools/screenshot/screenshot.cpp
                              tools/screenshot/capture_manifest.cpp
                              tools/screenshot/overlay_text.cpp)
                              # Spells a `shared/...` include, which resolves from `frontends`.
                              target_include_directories(screenshot PRIVATE frontends)
    target_include_directories(screenshot PRIVATE src tools/screenshot)
    target_link_libraries(screenshot PRIVATE prosper_core prosper_live_renderer Vulkan::Vulkan
                                              windowscodecs ole32)
  else()
    find_package(ZLIB QUIET)
    if(ZLIB_FOUND)
      add_executable(screenshot tools/screenshot/screenshot.cpp
                                tools/screenshot/capture_manifest.cpp
                                tools/screenshot/overlay_text.cpp)
                                # Spells a `shared/...` include, which resolves from `frontends`.
                                target_include_directories(screenshot PRIVATE frontends)
      target_include_directories(screenshot PRIVATE src tools/screenshot)
      target_link_libraries(screenshot PRIVATE prosper_core prosper_live_renderer Vulkan::Vulkan ZLIB::ZLIB)
    else()
      message(STATUS "zlib not found — skipping the screenshot tool")
    endif()
  endif()
endif()

# Optional FFmpeg AJM codec backend (#1347). Keep it outside prosper_core so core-only/headless
# builds retain no host codec dependency. The core owns the AJM ABI and copies; this frontend only
# accepts host byte spans and returns interleaved S16 PCM. MP3 is the first implemented codec.
option(PROSPER_AUDIO_FFMPEG "Build the FFmpeg AJM compressed-audio decoder backend" ON)
if(UNIX AND PROSPER_AUDIO_FFMPEG AND TARGET prosper_core)
  find_package(PkgConfig QUIET)
  if(PkgConfig_FOUND)
    pkg_check_modules(PROSPER_AJM_FFMPEG QUIET IMPORTED_TARGET
                      libavcodec libavutil libswresample)
  endif()
  # On Apple Silicon the macOS build is x86_64 (Rosetta), while Homebrew's FFmpeg is arm64-only:
  # pkg-config finds it, and the build then fails at the final link with undefined _swr_init and
  # friends. Ask the linker before adopting it, so a mismatched FFmpeg is reported here and skipped
  # rather than breaking every executable that links the backend.
  set(_prosper_ffmpeg_wrong_arch FALSE)
  if(PROSPER_AJM_FFMPEG_FOUND AND APPLE)
    include(CheckCXXSourceCompiles)
    # Re-probe on every configure: the verdict depends on CMAKE_OSX_ARCHITECTURES and on which FFmpeg
    # is installed, and a cached result would survive a change to either.
    unset(PROSPER_AJM_FFMPEG_LINKS_FOR_TARGET_ARCH CACHE)
    set(_prosper_saved_required_includes "${CMAKE_REQUIRED_INCLUDES}")
    set(_prosper_saved_required_libraries "${CMAKE_REQUIRED_LIBRARIES}")
    set(CMAKE_REQUIRED_INCLUDES ${PROSPER_AJM_FFMPEG_INCLUDE_DIRS})
    set(CMAKE_REQUIRED_LIBRARIES ${PROSPER_AJM_FFMPEG_LINK_LIBRARIES})
    check_cxx_source_compiles([=[
      extern "C" {
      #include <libavcodec/avcodec.h>
      #include <libswresample/swresample.h>
      }
      int main() { return avcodec_version() != 0 && swr_alloc() != nullptr ? 0 : 1; }
    ]=] PROSPER_AJM_FFMPEG_LINKS_FOR_TARGET_ARCH)
    set(CMAKE_REQUIRED_INCLUDES "${_prosper_saved_required_includes}")
    set(CMAKE_REQUIRED_LIBRARIES "${_prosper_saved_required_libraries}")
    if(NOT PROSPER_AJM_FFMPEG_LINKS_FOR_TARGET_ARCH)
      set(_prosper_ffmpeg_arch "${CMAKE_OSX_ARCHITECTURES}")
      if(NOT _prosper_ffmpeg_arch)
        set(_prosper_ffmpeg_arch "${CMAKE_SYSTEM_PROCESSOR}")
      endif()
      list(GET PROSPER_AJM_FFMPEG_LINK_LIBRARIES 0 _prosper_ffmpeg_first_lib)
      message(WARNING "FFmpeg (${_prosper_ffmpeg_first_lib}) does not link for "
                      "${_prosper_ffmpeg_arch} (an arm64-only Homebrew FFmpeg cannot serve the "
                      "x86_64 Rosetta build) - AJM MP3/Opus decoding disabled")
      set(PROSPER_AJM_FFMPEG_FOUND FALSE)
      set(_prosper_ffmpeg_wrong_arch TRUE)
    endif()
  endif()
  if(PROSPER_AJM_FFMPEG_FOUND)
    add_library(prosper_ajm_ffmpeg STATIC
                frontends/audio_ffmpeg/ajm_ffmpeg.cpp)
    target_include_directories(prosper_ajm_ffmpeg PUBLIC frontends/audio_ffmpeg PRIVATE src)
    target_link_libraries(prosper_ajm_ffmpeg PUBLIC prosper_core
                          PkgConfig::PROSPER_AJM_FFMPEG)

    # Does not link prosper_core, so it does not inherit its PUBLIC `src`.
    prosper_add_gtest(test_ajm_mp3
      SOURCES tests/hle/audio/test_ajm_mp3.cpp
      INCLUDES src
      LIBRARIES prosper_ajm_ffmpeg)

    prosper_add_gtest(test_ajm_opus
      SOURCES tests/hle/audio/test_ajm_opus.cpp
      INCLUDES src tests/hle/audio
      LIBRARIES prosper_ajm_ffmpeg)

    # libSceAudiodec (UE Electra movie audio). Same backend, different guest-facing library.
    prosper_add_gtest(test_audiodec
      SOURCES tests/hle/audio/test_audiodec.cpp
      INCLUDES src
      LIBRARIES prosper_ajm_ffmpeg)

    if(TARGET boot_trace)
      target_link_libraries(boot_trace PRIVATE prosper_ajm_ffmpeg)
      target_compile_definitions(boot_trace PRIVATE PROSPER_AUDIO_FFMPEG=1)
    endif()
    if(TARGET screenshot)
      target_link_libraries(screenshot PRIVATE prosper_ajm_ffmpeg)
      target_compile_definitions(screenshot PRIVATE PROSPER_AUDIO_FFMPEG=1)
    endif()
  else()
    if(NOT _prosper_ffmpeg_wrong_arch)   # the probe above already explained a wrong-arch skip
      message(WARNING "FFmpeg audio development packages not found - AJM MP3 decoding disabled")
    endif()
  endif()
endif()

# Windows native AvPlayer backend (#841): Media Foundation performs MP4 demux and video/audio
# decode on a worker thread. Keep it outside prosper_core so core-only/headless builds retain no OS
# media dependency and so software fallback can never be selected merely because no frontend exists.
if(WIN32 AND TARGET prosper_core)
  add_library(prosper_video_mf STATIC
              frontends/video_mf/media_foundation_backend.cpp
              frontends/video_mf/mf_access_unit_decoder.cpp)
  target_include_directories(prosper_video_mf PUBLIC frontends/video_mf PRIVATE src)
  target_link_libraries(prosper_video_mf PUBLIC prosper_core
                         mfplat mfreadwrite mfuuid ole32 propsys d3d11 dxgi strmiids)

  add_executable(test_video_mf frontends/video_mf/test_video_mf.cpp)
  target_link_libraries(test_video_mf PRIVATE prosper_video_mf)
  # sceVideodec2's access-unit path (#2983) decodes committed elementary streams with no container
  # around the units: the Annex-B H.264 asset the Linux test uses, a second H.264 stream WITH
  # B-frames and an SPS crop (so display order and the display aperture are tested), and a VP9 IVF (libvpx, two-pass, with
  # hidden alt-ref frames packed into superframes). Expected per-frame hashes for all three come
  # from an independent host `ffmpeg` decode.
  target_compile_definitions(test_video_mf PRIVATE
    PROSPER_TEST_H264_AU_ASSET="${CMAKE_CURRENT_SOURCE_DIR}/tests/assets/h264_annexb_testpattern.264"
    PROSPER_TEST_VP9_AU_ASSET="${CMAKE_CURRENT_SOURCE_DIR}/tests/assets/vp9_testpattern.ivf"
    PROSPER_TEST_H264_BFRAMES_AU_ASSET="${CMAKE_CURRENT_SOURCE_DIR}/tests/assets/h264_annexb_bframes.264")
  add_test(NAME video_media_foundation COMMAND test_video_mf)
  # RUN_SERIAL: under the first parallel (`-j4`) CI run its forced-software H.264 pass produced 11
  # pictures from 12 access units, after passing on every serial run. That looks like a real
  # scheduling dependence in the Media Foundation drain (#4122), not a test artefact -- serial keeps
  # CI from reddening every lane while it is diagnosed, and is NOT a fix for it.
  set_tests_properties(video_media_foundation PROPERTIES RUN_SERIAL TRUE)

  # The join test_video_mf and test_videodec2_decode each stop short of: the guest's own Videodec2
  # lifecycle (CreateDecoder with codec 2382845, Decode, DeleteDecoder) on the real Media Foundation
  # backend, with every guest picture hashed against the ffmpeg reference (#2983).
  add_executable(test_videodec2_mf frontends/video_mf/test_videodec2_mf.cpp)
  target_link_libraries(test_videodec2_mf PRIVATE prosper_video_mf)
  target_include_directories(test_videodec2_mf PRIVATE tests)
  target_compile_definitions(test_videodec2_mf PRIVATE
    PROSPER_TEST_VP9_AU_ASSET="${CMAKE_CURRENT_SOURCE_DIR}/tests/assets/vp9_testpattern.ivf")
  add_test(NAME videodec2_media_foundation COMMAND test_videodec2_mf)

  if(TARGET boot_trace)
    target_link_libraries(boot_trace PRIVATE prosper_video_mf)
    target_compile_definitions(boot_trace PRIVATE PROSPER_VIDEO_MF=1)
  endif()

  # The capture frontend must boot with the same real decoder as boot_trace and prosper-app;
  # otherwise video-gated titles never reach the frames that the tool is meant to preserve.
  if(TARGET screenshot)
    target_link_libraries(screenshot PRIVATE prosper_video_mf)
    target_compile_definitions(screenshot PRIVATE PROSPER_VIDEO_MF=1)
  endif()
endif()

# Linux native AvPlayer backend (#841): FFmpeg provides MP4 demux/audio conversion and drives a
# VA-API hardware decoder for video. Keep the dependency outside prosper_core; systems without the
# development packages retain the headless core but do not advertise a native video backend.
option(PROSPER_VIDEO_VAAPI "Build the Linux FFmpeg/VA-API AvPlayer backend" ON)
if(UNIX AND NOT APPLE AND PROSPER_VIDEO_VAAPI AND TARGET prosper_core)
  find_package(PkgConfig QUIET)
  if(PkgConfig_FOUND)
    pkg_check_modules(PROSPER_FFMPEG QUIET IMPORTED_TARGET
                      libavformat libavcodec libavutil libswresample libswscale libva)
  endif()
  if(PROSPER_FFMPEG_FOUND)
    add_library(prosper_video_vaapi STATIC
                frontends/video_vaapi/vaapi_backend.cpp)
    target_include_directories(prosper_video_vaapi PUBLIC frontends/video_vaapi PRIVATE src)
    target_link_libraries(prosper_video_vaapi PUBLIC prosper_core PkgConfig::PROSPER_FFMPEG)

    add_executable(test_video_vaapi frontends/video_vaapi/test_video_vaapi.cpp)
    target_link_libraries(test_video_vaapi PRIVATE prosper_video_vaapi)
    # Point the test at a committed test-pattern clip so it decodes a real stream. On headless CI
    # (no VA-API hardware) a successful decode exercises the default software-decode fallback (#320).
    # The second asset is a raw Annex-B H.264 ELEMENTARY STREAM, not a container: sceVideodec2's
    # caller demuxes itself and submits access units, so the AU-decode section (#2270) needs a
    # stream with no container around it, framed by access-unit delimiters the way a live guest's
    # is. Its expected per-frame hashes come from an independent host `ffmpeg` decode.
    target_compile_definitions(test_video_vaapi PRIVATE
      PROSPER_TEST_VIDEO_ASSET="${CMAKE_CURRENT_SOURCE_DIR}/tests/assets/testpattern.mp4"
      PROSPER_TEST_H264_AU_ASSET="${CMAKE_CURRENT_SOURCE_DIR}/tests/assets/h264_annexb_testpattern.264")
    add_test(NAME video_vaapi COMMAND test_video_vaapi)

    if(TARGET boot_trace)
      target_link_libraries(boot_trace PRIVATE prosper_video_vaapi)
      target_compile_definitions(boot_trace PRIVATE PROSPER_VIDEO_VAAPI=1)
    endif()

    # Match boot_trace and prosper-app: headless captures must install the real Linux decoder too,
    # otherwise video-gated titles only expose an empty scanout to the screenshot frontend.
    if(TARGET screenshot)
      target_link_libraries(screenshot PRIVATE prosper_video_vaapi)
      target_compile_definitions(screenshot PRIVATE PROSPER_VIDEO_VAAPI=1)
    endif()
  else()
    message(WARNING "FFmpeg/libva development packages not found - Linux AvPlayer video disabled")
  endif()
endif()

# macOS native AvPlayer backend: AVFoundation demuxes, VideoToolbox decodes in hardware. Outside
# prosper_core for the same reasons as the Media Foundation and VA-API backends above: core-only and
# headless builds carry no OS media dependency, and the app installs it explicitly.
option(PROSPER_VIDEO_VT "Build the macOS AVFoundation/VideoToolbox AvPlayer backend" ON)
if(APPLE AND PROSPER_VIDEO_VT AND TARGET prosper_core)
  enable_language(OBJCXX)
  add_library(prosper_video_vt STATIC frontends/video_vt/videotoolbox_backend.mm)
  target_include_directories(prosper_video_vt PUBLIC frontends/video_vt PRIVATE src)
  target_compile_options(prosper_video_vt PRIVATE -fobjc-arc)
  # CMAKE_CXX_STANDARD does not apply to OBJCXX; pin it so the .mm does not ride the compiler default.
  set_target_properties(prosper_video_vt PROPERTIES OBJCXX_STANDARD 20 OBJCXX_STANDARD_REQUIRED ON)
  target_link_libraries(prosper_video_vt PUBLIC prosper_core
                        "-framework AVFoundation" "-framework CoreMedia" "-framework CoreVideo"
                        "-framework VideoToolbox" "-framework Foundation")

  prosper_add_gtest(test_video_vt
    SOURCES frontends/video_vt/test_video_vt.cpp
    LIBRARIES prosper_video_vt)
  # H.264 Main + AAC stereo, the codec pair of Silksong's cinematics; testpattern.mp4 is MPEG-4
  # Part 2, which VideoToolbox has no hardware decoder for.
  target_compile_definitions(test_video_vt PRIVATE
    PROSPER_TEST_H264_AAC_ASSET="${CMAKE_CURRENT_SOURCE_DIR}/tests/assets/h264_aac_testpattern.mp4"
    PROSPER_TEST_H264_CROP_ASSET="${CMAKE_CURRENT_SOURCE_DIR}/tests/assets/h264_crop_testpattern.mp4"
    PROSPER_TEST_MPEG4_ASSET="${CMAKE_CURRENT_SOURCE_DIR}/tests/assets/testpattern.mp4")

  if(TARGET boot_trace)
    target_link_libraries(boot_trace PRIVATE prosper_video_vt)
    target_compile_definitions(boot_trace PRIVATE PROSPER_VIDEO_VT=1)
  endif()
  if(TARGET screenshot)
    target_link_libraries(screenshot PRIVATE prosper_video_vt)
    target_compile_definitions(screenshot PRIVATE PROSPER_VIDEO_VT=1)
  endif()
endif()

# --- SDL3 frontends (optional; headless core stays dependency-free) --------------------------
# The sceAudioOut sink (frontends/audio_sdl3) and the libScePad gamepad backend (frontends/pad_sdl3)
# can both use SDL3. They must share ONE SDL3 build: if each fetched independently, the first fetch's
# option set would win — e.g. an AUDIO+PAD build would get audio's joystick-less SDL and the gamepad
# backend would silently have no device. So acquire SDL3 ONCE here, with the UNION of the subsystems
# the enabled frontends need (JOYSTICK/HIDAPI only when the pad frontend is on). Both consumer blocks
# below just use the resulting SDL3::SDL3 target.
option(PROSPER_APP        "Build the prosper-app frontend (SDL3 window + Vulkan present)" OFF)
# The app frontend defaults to having audio, because it is the frontend a USER runs and a silent
# one is indistinguishable from broken audio. With PROSPER_AUDIO_SDL3 off, install_sdl3_audio_sink()
# is compiled out of prosper-app entirely and NOTHING is logged about it -- GTA V appeared to have
# no audio for exactly this reason, while the guest was publishing 99.9%-non-zero PCM that prosper
# mixed correctly into a sink with no device behind it (#2402).
#
# option() honours an existing cache entry, so an explicit -DPROSPER_AUDIO_SDL3=OFF still wins;
# this only changes the DEFAULT, and only when the app is being built.
option(PROSPER_AUDIO_SDL3 "Build the optional SDL3 audio frontend"   ${PROSPER_APP})
option(PROSPER_PAD_SDL3   "Build the optional SDL3 gamepad frontend" OFF)
if((PROSPER_AUDIO_SDL3 OR PROSPER_PAD_SDL3 OR PROSPER_APP) AND TARGET prosper_core AND NOT TARGET SDL3::SDL3)
  # On macOS the build is x86_64 (Rosetta), but a system SDL3 (e.g. Homebrew) is arm64-only and won't
  # link — so skip discovery there and always FetchContent+build SDL3 for the requested architecture.
  if(NOT APPLE)
    find_package(SDL3 QUIET CONFIG)
  endif()
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
      # Wayland/X11 are the Linux window backends; on macOS SDL uses Cocoa (forcing X11/Wayland ON
      # there makes SDL hunt for X11/Wayland dev headers that don't exist). Only request them on Linux.
      if(NOT APPLE)
        set(SDL_WAYLAND ON CACHE BOOL "" FORCE)
        set(SDL_X11     ON CACHE BOOL "" FORCE)
      endif()
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
    # The app frontend opens a title with the host's native folder picker (#1469); the audio- and
    # pad-only builds have no window to parent a dialog to, so they keep SDL's dialog code out.
    if(PROSPER_APP)
      set(SDL_DIALOG ON CACHE BOOL "" FORCE)
    else()
      set(SDL_DIALOG OFF CACHE BOOL "" FORCE)
    endif()
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
# window. A SEPARATE Vulkan context from the core's headless render device (docs/subsystems/FRONTEND_APP.md);
# frames cross as CPU pixels. Entirely outside prosper_core. Default OFF; needs SDL3 (video) + Vulkan.
if(PROSPER_APP AND TARGET SDL3::SDL3)
  if(NOT TARGET Vulkan::Vulkan)
    find_package(Vulkan QUIET)
  endif()
  if(Vulkan_FOUND)
    # Dear ImGui for the game library view (#1471), vendored under third_party/imgui. FRONTEND ONLY:
    # prosper_core neither links nor sees it, so the one-way dependency arrow in docs/subsystems/FRONTEND_APP.md
    # is unchanged. Built as its own static lib so the vendored sources compile with their own
    # warning expectations rather than prosper's.
    add_library(prosper_imgui STATIC
                third_party/imgui/src/imgui.cpp
                third_party/imgui/src/imgui_draw.cpp
                third_party/imgui/src/imgui_tables.cpp
                third_party/imgui/src/imgui_widgets.cpp
                third_party/imgui/backends/imgui_impl_sdl3.cpp
                third_party/imgui/backends/imgui_impl_vulkan.cpp)
    target_include_directories(prosper_imgui PUBLIC third_party/imgui/src third_party/imgui/backends)
    target_link_libraries(prosper_imgui PUBLIC SDL3::SDL3 Vulkan::Vulkan)
    # Vendored third-party code is not held to prosper's warning settings; keep its noise out of our
    # build output so a real warning in prosper's own code stays visible.
    if(NOT MSVC)
      target_compile_options(prosper_imgui PRIVATE -w)
    endif()

    add_executable(prosper-app frontends/prosper-app/main.cpp frontends/prosper-app/library_ui.cpp
                               frontends/prosper-app/library_media.cpp
                               frontends/prosper-app/fps_overlay.cpp)
    # third_party/renderdoc: the vendored in-application API header (#3321). Header-only and
    # never linked -- librenderdoc.so is dlopen'd at runtime by the capture trigger.
    target_include_directories(prosper-app PRIVATE src third_party/stb third_party/renderdoc
                                                   frontends)
    target_link_libraries(prosper-app PRIVATE prosper_core SDL3::SDL3 Vulkan::Vulkan prosper_imgui
                                              prosper_performance_capture)
    target_compile_definitions(prosper-app PRIVATE PROSPER_HAVE_LIBRARY_UI=1)
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
# The sceAudioOut HLE is backend-agnostic (src/hle/audio/audio.hpp). This builds an SDL3-backed sink,
# entirely OUTSIDE prosper_core, so the guest's audio plays on the host.
if(PROSPER_AUDIO_SDL3 AND TARGET SDL3::SDL3)
  add_library(prosper_audio_sdl3 STATIC frontends/audio_sdl3/audio_sink_sdl3.cpp)
  target_include_directories(prosper_audio_sdl3 PUBLIC frontends/audio_sdl3 src)
  target_link_libraries(prosper_audio_sdl3 PUBLIC prosper_core SDL3::SDL3)
  target_compile_definitions(prosper_audio_sdl3 PUBLIC PROSPER_AUDIO_SDL3=1)

  # Headless self-test of the full SDL3 path (run under the SDL "dummy" audio driver in CI):
  # install the sink, drive the sceAudioOut HLE, assert grains flow through SDL with no device.
  add_executable(test_audio_sdl3 frontends/audio_sdl3/test_audio_sdl3.cpp)
  # Does not link prosper_core, so it does not inherit its PUBLIC `src`.
  target_include_directories(test_audio_sdl3 PRIVATE src)
  target_link_libraries(test_audio_sdl3 PRIVATE prosper_audio_sdl3)
  add_test(NAME audio_sdl3 COMMAND test_audio_sdl3)
  set_tests_properties(audio_sdl3 PROPERTIES ENVIRONMENT "SDL_AUDIO_DRIVER=dummy")
  add_executable(test_audio_demand frontends/audio_sdl3/test_audio_demand.cpp)
  target_link_libraries(test_audio_demand PRIVATE prosper_audio_sdl3)
  add_test(NAME audio_demand COMMAND test_audio_demand)
  set_tests_properties(audio_demand PROPERTIES ENVIRONMENT "SDL_AUDIO_DRIVER=dummy")
  add_test(NAME audio_lifecycle_destroy_race COMMAND test_audio_demand lifecycle-race)
  set_tests_properties(audio_lifecycle_destroy_race PROPERTIES ENVIRONMENT "SDL_AUDIO_DRIVER=dummy" TIMEOUT 15)
  add_test(NAME audio_lifecycle_sink COMMAND test_audio_demand lifecycle-sink)
  set_tests_properties(audio_lifecycle_sink PROPERTIES
    ENVIRONMENT "SDL_AUDIO_DRIVER=dummy;PROSPER_AUDIO_DEMAND=1;PROSPER_AUDIO_LIFECYCLE=1")
  add_test(NAME audio_lifecycle_sink_disabled COMMAND test_audio_demand lifecycle-sink)
  set_tests_properties(audio_lifecycle_sink_disabled PROPERTIES
    ENVIRONMENT "SDL_AUDIO_DRIVER=dummy;PROSPER_AUDIO_DEMAND=1;PROSPER_AUDIO_LIFECYCLE=0")
  add_test(NAME audio_demand_return_open COMMAND test_audio_demand return-open)
  add_test(NAME audio_demand_ordered_quit COMMAND test_audio_demand ordered-quit)
  set_tests_properties(audio_demand_return_open audio_demand_ordered_quit PROPERTIES
    ENVIRONMENT "SDL_AUDIO_DRIVER=dummy;PROSPER_AUDIO_DEMAND=1")
  add_test(NAME audio_demand_external_quit COMMAND test_audio_demand external-quit)
  set_tests_properties(audio_demand_external_quit PROPERTIES
    ENVIRONMENT "SDL_AUDIO_DRIVER=dummy;PROSPER_AUDIO_DEMAND=0;PROSPER_AUDIO_QUEUE_TIMELINE=0")

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
  if(TARGET prosper_ajm_ffmpeg)
    target_link_libraries(prosper-app PRIVATE prosper_ajm_ffmpeg)
    target_compile_definitions(prosper-app PRIVATE PROSPER_AUDIO_FFMPEG=1)
    message(STATUS "prosper-app: FFmpeg AJM audio decoding enabled")
  endif()
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

if(TARGET prosper-app AND TARGET prosper_video_mf)
  target_link_libraries(prosper-app PRIVATE prosper_video_mf)
  target_compile_definitions(prosper-app PRIVATE PROSPER_VIDEO_MF=1)
  message(STATUS "prosper-app: Media Foundation video enabled")
endif()

if(TARGET prosper-app AND TARGET prosper_video_vt)
  target_link_libraries(prosper-app PRIVATE prosper_video_vt)
  target_compile_definitions(prosper-app PRIVATE PROSPER_VIDEO_VT=1)
  message(STATUS "prosper-app: AVFoundation/VideoToolbox video enabled")
endif()

if(TARGET prosper-app AND TARGET prosper_video_vaapi)
  target_link_libraries(prosper-app PRIVATE prosper_video_vaapi)
  target_compile_definitions(prosper-app PRIVATE PROSPER_VIDEO_VAAPI=1)
  message(STATUS "prosper-app: FFmpeg/VA-API video enabled")
endif()

# Image robustness (#3531): the device-side half of the recompiler's out-of-bounds image-read
# contract, and the refusal that replaces executing it undefined. Registered OUTSIDE the per-platform
# blocks because the contract is not platform-specific and the bug it guards was a divergence between
# two device paths -- a guard built on only one host would have the same blind spot. Queries a
# physical device but creates none, so it runs wherever an ICD exists and reports a skip (never a
# silent pass) where one does not.
if(TARGET Vulkan::Vulkan AND TARGET prosper_core)
  prosper_add_gtest(test_fragment_draw_plan
    SOURCES tests/gpu/execute/test_fragment_draw_plan.cpp
    INCLUDES tests frontends LIBRARIES prosper_core Vulkan::Vulkan LABELS unit PROPERTIES TIMEOUT 120)
  prosper_add_gtest(test_fragment_draw_exec
    SOURCES tests/gpu/execute/test_fragment_draw_exec.cpp
    INCLUDES tests frontends LIBRARIES prosper_core Vulkan::Vulkan LABELS gpu PROPERTIES TIMEOUT 120)
  prosper_add_gtest(test_ps_pull_model_exec
    SOURCES tests/gpu/execute/test_ps_pull_model_exec.cpp
    INCLUDES tests frontends LIBRARIES prosper_core Vulkan::Vulkan LABELS gpu PROPERTIES TIMEOUT 120)
  prosper_add_gtest(test_fragment_entry_prefix_exec
    SOURCES tests/gpu/execute/test_fragment_entry_prefix_exec.cpp
    INCLUDES tests
    LIBRARIES prosper_core Vulkan::Vulkan
    LABELS gpu
    PROPERTIES TIMEOUT 120)
  prosper_add_gtest(test_fragment_packet_definedness_exec
    SOURCES tests/gpu/execute/test_fragment_packet_definedness_exec.cpp
    INCLUDES tests LIBRARIES prosper_core Vulkan::Vulkan LABELS gpu PROPERTIES TIMEOUT 120)
  prosper_add_gtest(test_fragment_packet_raw_masks_exec
    SOURCES tests/gpu/execute/test_fragment_packet_raw_masks_exec.cpp
    INCLUDES tests LIBRARIES prosper_core Vulkan::Vulkan LABELS gpu PROPERTIES TIMEOUT 120)
  prosper_add_gtest(test_fragment_packet_mask_entry_exec
    SOURCES tests/gpu/execute/test_fragment_packet_mask_entry_exec.cpp
    INCLUDES tests LIBRARIES prosper_core Vulkan::Vulkan LABELS gpu PROPERTIES TIMEOUT 120)

  add_executable(test_fragment_packet_exec tests/gpu/execute/test_fragment_packet_exec.cpp)
  target_include_directories(test_fragment_packet_exec PRIVATE tests)
  target_link_libraries(test_fragment_packet_exec PRIVATE prosper_core Vulkan::Vulkan)
  add_test(NAME fragment_packet_exec COMMAND test_fragment_packet_exec)
  set_tests_properties(fragment_packet_exec PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 120)

  prosper_add_gtest(test_fragment_resource_packet_exec
    SOURCES tests/gpu/execute/test_fragment_resource_packet_exec.cpp
    INCLUDES tests LIBRARIES prosper_core Vulkan::Vulkan
    PROPERTIES TIMEOUT 120 SKIP_REGULAR_EXPRESSION ".  SKIPPED ")
  prosper_add_gtest(test_fragment_special_f32_exec
    SOURCES tests/gpu/execute/test_fragment_special_f32_exec.cpp
    INCLUDES tests LIBRARIES prosper_core Vulkan::Vulkan
    PROPERTIES TIMEOUT 120 SKIP_REGULAR_EXPRESSION ".  SKIPPED ")
  prosper_add_gtest(test_fragment_packet_wave_data_exec
    SOURCES tests/gpu/execute/test_fragment_packet_wave_data_exec.cpp
    INCLUDES tests LIBRARIES prosper_core Vulkan::Vulkan
    PROPERTIES TIMEOUT 120 SKIP_REGULAR_EXPRESSION ".  SKIPPED ")
  prosper_add_gtest(test_fragment_packet_exports_exec
    SOURCES tests/gpu/execute/test_fragment_packet_exports_exec.cpp
    INCLUDES tests LIBRARIES prosper_core Vulkan::Vulkan
    PROPERTIES TIMEOUT 120 SKIP_REGULAR_EXPRESSION ".  SKIPPED ")

  add_executable(test_float_transport_render tests/gpu/execute/test_float_transport_render.cpp)
  target_include_directories(test_float_transport_render PRIVATE tests frontends)
  target_link_libraries(test_float_transport_render PRIVATE prosper_core Vulkan::Vulkan)
  add_test(NAME float_transport_pixel_oracle COMMAND test_float_transport_render --oracle-only)
  set_tests_properties(float_transport_pixel_oracle PROPERTIES TIMEOUT 10)
  add_test(NAME float_transport_render COMMAND test_float_transport_render)
  set_tests_properties(float_transport_render PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 60)

  # Spells `shared/...` and `gpu/...` includes, which resolve from `frontends` and `src`.
  prosper_add_gtest(test_image_robustness
    SOURCES tests/shared/device/test_image_robustness.cpp
    INCLUDES frontends src
    LIBRARIES prosper_core Vulkan::Vulkan)
endif()

# Shared-device adoption guard (#1091). prosper_live_renderer only exists in frontend-enabled
# configurations and is defined later in this file than the core test block, so register this
# after the fact and only when the target is actually present.
if(TARGET prosper_live_renderer AND TARGET prosper_core)
  # Includes a shared fixture, which resolves from the `tests` root.
  # Spells a `shared/...` include, which resolves from `frontends`.
  prosper_add_gtest(test_shared_vulkan_device
    SOURCES tests/shared/live/test_shared_vulkan_device.cpp
    INCLUDES tests frontends
    LIBRARIES prosper_live_renderer prosper_core)

  # Live-target format round trip: every LiveTargetPixelFormat a compute dispatch can import must
  # survive the write-back notifier. An unnamed format is reported as RGBA8, fails the mirror
  # identity check, and silently leaves the renderer's target invalidated (#773's family).
  # Spells a `shared/...` include, which resolves from `frontends`.
  prosper_add_gtest(test_live_target_format
    SOURCES tests/shared/live/test_live_target_format.cpp
    INCLUDES frontends src tests
    LIBRARIES prosper_live_renderer prosper_core Vulkan::Vulkan)

  add_executable(test_guest_write_drain tests/shared/live/test_guest_write_drain.cpp)
  target_include_directories(test_guest_write_drain PRIVATE frontends src tests)
  target_link_libraries(test_guest_write_drain PRIVATE prosper_live_renderer prosper_core Vulkan::Vulkan)
  add_test(NAME guest_write_drain COMMAND test_guest_write_drain)
  add_test(NAME guest_write_drain_ordinary COMMAND test_guest_write_drain --ordinary)
  add_test(NAME guest_write_drain_watch COMMAND test_guest_write_drain --ordinary)
  set_tests_properties(guest_write_drain PROPERTIES
    ENVIRONMENT "PROSPER_GPU_REPLAY_RTT_SEEDS=1;PROSPER_WRITE_DRAIN_CENSUS=1" TIMEOUT 60)
  set_tests_properties(guest_write_drain_ordinary PROPERTIES
    ENVIRONMENT "PROSPER_GPU_REPLAY_RTT_SEEDS=1;PROSPER_WRITE_DRAIN_CENSUS=1;PROSPER_NO_RTT_WRITE_BATCH=1" TIMEOUT 60)
  set_tests_properties(guest_write_drain_watch PROPERTIES
    ENVIRONMENT "PROSPER_GPU_REPLAY_RTT_SEEDS=1;PROSPER_WRITE_DRAIN_CENSUS=1;PROSPER_RTT_INVALIDATE_WATCH=0x10000" TIMEOUT 60)

  # GPU scanout handoff (#1270): the front-buffer image blits into a scanout slot byte-exactly, and the
  # producer/consumer slot handoff never tears or reuses an image in flight. Headless (no surface).
  # Spells a `shared/...` include, which resolves from `frontends`.
  prosper_add_gtest(test_present_blit
    SOURCES tests/shared/present/test_present_blit.cpp
    INCLUDES frontends src tests
    LIBRARIES prosper_live_renderer prosper_core Vulkan::Vulkan)

  # #3915: a display buffer written by a compute storage store is presented through GPU present from
  # a mirror of the dispatch's own result, byte-identical to the CPU fallback, and never after the
  # guest bytes change. Real device, real HLE VideoOut registration, real guest write watch.
  add_executable(test_compute_scanout tests/shared/present/test_compute_scanout.cpp)
  target_include_directories(test_compute_scanout PRIVATE frontends src tests)
  target_link_libraries(test_compute_scanout PRIVATE prosper_live_renderer prosper_core Vulkan::Vulkan)
  add_test(NAME compute_scanout_present COMMAND test_compute_scanout)
  set_tests_properties(compute_scanout_present PROPERTIES TIMEOUT 120)
  add_test(NAME present_blit_trace COMMAND test_present_blit)
  set_tests_properties(present_blit_trace PROPERTIES
    ENVIRONMENT "PROSPER_PRESENT_HANDOFF_TRACE=1")
endif()

# #3372: gpu_replay's two graphics shader dumps produce OPPOSITE things -- --dump-shader writes
# prosper's recompiled SPIR-V, --dump-shader-raw writes the guest RDNA2 that fed it -- and the
# flag names alone do not say which way round. An investigation read the RDNA2 out of the flag
# whose name suggested SPIR-V and lost a session to it. The usage legend is the fix, so something
# has to fail when the legend is dropped; nothing else in the tree can. Behavioural rather than a
# source grep: it runs the real binary and reads what an operator would actually see.
if(TARGET gpu_replay AND Python3_Interpreter_FOUND)
  add_test(NAME gpu_replay_float_transport_cli
           COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/gpu_replay/test_float_transport.py")
  set_tests_properties(gpu_replay_float_transport_cli PROPERTIES TIMEOUT 300 SKIP_RETURN_CODE 77)
  set_property(TEST gpu_replay_float_transport_cli APPEND PROPERTY ENVIRONMENT
    "PROSPER_GPU_REPLAY_BIN=$<TARGET_FILE:gpu_replay>"
    "PROSPER_FLOAT_TRANSPORT_FIXTURE_BIN=$<TARGET_FILE:test_float_transport_capture>")
  add_test(NAME gpu_replay_fragment_float_mode_cli
           COMMAND "${Python3_EXECUTABLE}"
                   "${CMAKE_SOURCE_DIR}/tools/gpu_replay/test_fragment_float_mode.py")
  set_tests_properties(gpu_replay_fragment_float_mode_cli PROPERTIES TIMEOUT 300 SKIP_RETURN_CODE 77)
  set_property(TEST gpu_replay_fragment_float_mode_cli APPEND PROPERTY ENVIRONMENT
    "PROSPER_GPU_REPLAY_BIN=$<TARGET_FILE:gpu_replay>"
    "PROSPER_FRAGMENT_FLOAT_MODE_FIXTURE_BIN=$<TARGET_FILE:test_fragment_float_mode_capture>")
  add_test(NAME gpu_replay_fragment_width_cli
           COMMAND "${Python3_EXECUTABLE}"
                   "${CMAKE_SOURCE_DIR}/tools/gpu_replay/test_fragment_width.py")
  set_tests_properties(gpu_replay_fragment_width_cli PROPERTIES TIMEOUT 300 SKIP_RETURN_CODE 77)
  set_property(TEST gpu_replay_fragment_width_cli APPEND PROPERTY ENVIRONMENT
    "PROSPER_GPU_REPLAY_BIN=$<TARGET_FILE:gpu_replay>"
    "PROSPER_REPLAY_FRAGMENT_WIDTH_FIXTURE_BIN=$<TARGET_FILE:test_replay_fragment_width>")

  add_test(NAME gpu_replay_cli_contract
           COMMAND "${Python3_EXECUTABLE}"
                   "${CMAKE_SOURCE_DIR}/tools/gpu_replay/test_cli_contract.py")
  # SKIP_RETURN_CODE so "gpu_replay was not built" reports as (Skipped) by name rather than as a
  # pass -- the tool needs Vulkan and is genuinely absent in some environments.
  set_tests_properties(gpu_replay_cli_contract PROPERTIES TIMEOUT 300 SKIP_RETURN_CODE 77)
  set_property(TEST gpu_replay_cli_contract APPEND PROPERTY ENVIRONMENT
    "PROSPER_GPU_REPLAY_BIN=$<TARGET_FILE:gpu_replay>"
    "PROSPER_GPU_REPLAY_FIXTURE_BIN=$<TARGET_FILE:test_gpu_replay_shader_dump>")
  if(TARGET ngg_capture_probe)
    set_property(TEST gpu_replay_cli_contract APPEND PROPERTY ENVIRONMENT
      "PROSPER_NGG_CAPTURE_PROBE_BIN=$<TARGET_FILE:ngg_capture_probe>")
  endif()
endif()


# The common-v69 metadata controls launch the actual tool, so keep them behind its build gate.
if(TARGET gpu_replay)
  prosper_add_gtest(test_gpu_capture_observation_cli
    SOURCES tests/gpu/capture/test_gpu_capture_observation.cpp
    INCLUDES tests
    LIBRARIES prosper_core
    ENVIRONMENT "PROSPER_GPU_REPLAY_TEST_BINARY=$<TARGET_FILE:gpu_replay>"
    PROPERTIES TIMEOUT 60)
  target_compile_definitions(test_gpu_capture_observation_cli PRIVATE PROSPER_OBSERVATION_CLI_TESTS=1)
  add_dependencies(test_gpu_capture_observation_cli gpu_replay)
endif()

# CPU-only actual ordered-executor/backend-metadata contract, shared by Windows and UNIX. Links
# the normal renderer but deliberately creates no device or GPU work and needs no game dump.
if(TARGET prosper_live_renderer AND TARGET Vulkan::Vulkan)
  prosper_add_gtest(test_ordered_graphics_read_point
    SOURCES tests/shared/live/test_ordered_graphics_read_point.cpp
    INCLUDES frontends src tests
    LIBRARIES prosper_core prosper_live_renderer Vulkan::Vulkan
    LABELS unit PROPERTIES TIMEOUT 30)
endif()

# ---------------------------------------------------------------------------
# Per-case scratch directory for every registered test (#1613).
#
# Tests that must touch the real filesystem take their paths from
# tests/fixtures/test_scratch.h, which reads PROSPER_TEST_SCRATCH_DIR. Giving each ctest
# *case* its own root is what makes `ctest -j` safe: one test binary is
# sometimes registered as several cases (game_compute_exec and
# game_compute_exec_no_adaptive_storage_result are the same executable), so
# per-binary uniqueness is not enough. The helper adds a pid component
# underneath, which separates concurrent ctest invocations from different
# worktrees. The root is inside the build directory, never /tmp — see the
# header for why that matters on this machine.
#
# This must stay at the very end of the file: it snapshots the DIRECTORY TESTS
# property, so any add_test() below it would not be covered -- and neither is any
# GoogleTest case, because prosper_add_gtest discovers those at ctest time. A test
# that is missed still fails safe rather than silently — test_scratch.h falls back to
# <cwd>/prosper-test-scratch/, and ctest's working directory is this same build
# directory, so it lands off the tmpfs either way. It loses the per-case split: the
# pid component still separates concurrent processes, but they share one root, so
# test_scratch.h retries creation when a peer removes that root under it.
get_property(prosper_registered_tests DIRECTORY PROPERTY TESTS)
foreach(prosper_test_case IN LISTS prosper_registered_tests)
  set_property(TEST ${prosper_test_case} APPEND PROPERTY ENVIRONMENT
    "PROSPER_TEST_SCRATCH_DIR=${CMAKE_CURRENT_BINARY_DIR}/test-scratch/${prosper_test_case}")
endforeach()
