#include "present_mode.hpp"

#include <cstdio>

using prosper::frontend::AppPresentMode;
using prosper::frontend::parse_present_mode;
using prosper::frontend::present_mode_name;
using prosper::frontend::select_present_mode;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { std::printf("  [FAIL] %s\n", m); ++fails; } \
                         else       { std::printf("  [ok]   %s\n", m); } } while (0)

int main() {
    std::printf("== test_prosper_app_present_mode ==\n");

    AppPresentMode mode = AppPresentMode::fifo;
    CHECK(parse_present_mode("mailbox", mode) && mode == AppPresentMode::mailbox,
          "mailbox option parses");
    CHECK(parse_present_mode("immediate", mode) && mode == AppPresentMode::immediate,
          "immediate option parses");
    CHECK(parse_present_mode("fifo", mode) && mode == AppPresentMode::fifo,
          "fifo option parses");
    CHECK(!parse_present_mode("adaptive", mode) && mode == AppPresentMode::fifo,
          "unknown option is rejected without changing the selection");

    auto selected = select_present_mode(AppPresentMode::fifo, true, true);
    CHECK(selected.mode == AppPresentMode::fifo && !selected.fell_back,
          "FIFO remains selected even when optional modes are available");
    selected = select_present_mode(AppPresentMode::mailbox, true, false);
    CHECK(selected.mode == AppPresentMode::mailbox && !selected.fell_back,
          "supported mailbox mode is selected");
    selected = select_present_mode(AppPresentMode::mailbox, false, true);
    CHECK(selected.mode == AppPresentMode::fifo && selected.fell_back,
          "unsupported mailbox mode falls back to FIFO");
    selected = select_present_mode(AppPresentMode::immediate, false, true);
    CHECK(selected.mode == AppPresentMode::immediate && !selected.fell_back,
          "supported immediate mode is selected");
    selected = select_present_mode(AppPresentMode::immediate, true, false);
    CHECK(selected.mode == AppPresentMode::fifo && selected.fell_back,
          "unsupported immediate mode falls back to FIFO");
    CHECK(std::string_view(present_mode_name(AppPresentMode::mailbox)) == "mailbox",
          "selected modes have stable log names");

    if (fails) { std::printf("== FAIL: %d ==\n", fails); return 1; }
    std::printf("== PASS ==\n");
    return 0;
}
