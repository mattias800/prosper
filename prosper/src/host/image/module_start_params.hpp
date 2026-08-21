// module_start_params.hpp — which linked modules need a real SCE module-param descriptor.
//
// prosper preloads a title's PRXs itself and starts each one through the PS5 module-entry ABI,
// `module_start(size_t argc, const void *argp)`. Most modules ignore both arguments, so the default
// (0, NULL) is fine. A native plugin that performs a version handshake does NOT: it stores argc and
// argp at module_start and validates them later, and (0, NULL) makes it take its mismatch branch and
// then dereference the NULL argp.
//
// This is pure policy over the fixed guest base map in boot_program.hpp — no dump, no SELF parser,
// no filesystem — so both the "needs it" and the "must not get it" direction are checkable in
// ordinary CI. The second direction is the load-bearing one: a predicate that answered true for
// everything would satisfy the first arm perfectly while changing the entry ABI of every module
// prosper links.
#pragma once
#include <cstdint>
#include <utility>
#include <vector>

namespace prosper {

// Guest [begin, end) ranges whose module_start must receive the descriptor. See the .cpp for why
// each one is here.
std::vector<std::pair<uint64_t, uint64_t>> module_start_param_ranges();

// True when an init function at guest address `addr` falls in one of the ranges above.
bool module_start_wants_param_descriptor(uint64_t addr);

} // namespace prosper
