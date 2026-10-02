// The exit-status contract for a dead guest (#4140). Pure logic: no SDL, Vulkan or guest needed.
#include "guest_end_status.hpp"

#include <cstdio>
#include <string>

static int failures = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++failures; } } while (0)

int main() {
    using namespace prosper::app;
    CHECK(exit_status(-1, false) == 0);   // guest still running at shutdown
    CHECK(exit_status(0, false) == 0);    // clean return
    CHECK(exit_status(0, true) == 1);     // --frames shortfall, guest fine
    CHECK(exit_status(1, false) == kExitGuestFault);   // entry thread faulted
    CHECK(exit_status(2, true) == kExitGuestFault);    // fault outranks the shortfall
    CHECK(kExitGuestFault != 0 && kExitGuestFault != 1 && kExitGuestFault != 2);
    const std::string b = format_fault_banner("worker", 42, 0xC0000005u, 0x1234, "eboot+0x34");
    CHECK(b.find("worker thread tid=42") != std::string::npos);
    CHECK(b.find("code=0xc0000005") != std::string::npos);
    CHECK(b.find("eboot+0x34") != std::string::npos);
    CHECK(b.find("status 3") != std::string::npos);
    if (failures) return 1;
    std::puts("test_guest_end_status: OK");
    return 0;
}
