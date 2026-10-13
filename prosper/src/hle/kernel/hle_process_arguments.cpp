// Native libkernel's getargc/getargv return the cached count and the original guest startup argv
// vector. The host prepares that vector before module init; querying it never constructs host data.
#include "hle/dispatch/dispatch.hpp"
#include "host/image/process_entry.hpp"

namespace prosper {
namespace {
static PROSPER_SYSV_ABI uint64_t get_argc(uint64_t, uint64_t, uint64_t, uint64_t, uint64_t,
                                          uint64_t) {
    return process_entry_context().argc;
}
static PROSPER_SYSV_ABI uint64_t get_argv(uint64_t, uint64_t, uint64_t, uint64_t, uint64_t,
                                          uint64_t) {
    return process_entry_context().argv;
}
}   // namespace

void register_process_arguments_hle() {
    Hle::register_fn("iKJMWrAumPE", get_argc, "getargc");
    Hle::register_fn("FJmglmTMdr4", get_argv, "getargv");
}
}   // namespace prosper
