// Issue #3639: the guest FS base is restored from [handler_rsp+0x28]. If a new push changes the
// swap stub's frame, callbacks can restore the wrong base and corrupt guest TLS. Inspect bytes so
// this contract runs on hosts that cannot execute FSGSBASE.
#include "host/image/guest_fs_swap_stub.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace {

struct Layout {
    size_t saved_fs_offset{};
    size_t pushes_before_call{};
    size_t copied_stack_args{};
};

bool preserves_swap_stub_contract(const Layout& layout) {
    return layout.saved_fs_offset == 0x28 && layout.pushes_before_call == 5 &&
           layout.copied_stack_args == 4;
}

// Walk only the guest branch's push instructions and call. The emitted sequence is intentionally
// small and fixed: saved r11, four copied SysV stack arguments, then call r10.
bool inspect_guest_call(const std::vector<uint8_t>& code, size_t guest_begin, Layout& layout) {
    size_t p = guest_begin;
    size_t saved_fs_push = SIZE_MAX;
    size_t pushes = 0;
    while (p < code.size()) {
        if (p + 3 <= code.size() && code[p] == 0x41 && code[p + 1] == 0xFF && code[p + 2] == 0xD2) {
            if (saved_fs_push == SIZE_MAX) return false;
            layout.pushes_before_call = pushes;
            layout.saved_fs_offset = (pushes - saved_fs_push) * sizeof(uint64_t);
            return true;
        }
        size_t width = 0;
        bool push_instruction = true;
        if (p + 2 <= code.size() && code[p] == 0x41 && code[p + 1] == 0x53) {
            if (saved_fs_push == SIZE_MAX) saved_fs_push = pushes;
            width = 2;
        } else if (p + 4 <= code.size() && code[p] == 0xFF && code[p + 1] == 0x74 &&
                   code[p + 2] == 0x24) {
            width = 4;
            ++layout.copied_stack_args;
        } else if (code[p] >= 0x50 && code[p] <= 0x57) {
            width = 1;
        } else if (p + 2 <= code.size() && code[p] == 0x41 &&
                   code[p + 1] >= 0x50 && code[p + 1] <= 0x57) {
            width = 2;
        } else if (p + 7 <= code.size() && code[p] == 0x49 && code[p + 1] == 0x8B &&
                   code[p + 2] == 0x83) {
            width = 7; // load host FS base from saved host TLS
            push_instruction = false;
        } else if (p + 5 <= code.size() && code[p] == 0xF3 && code[p + 1] == 0x48 &&
                   code[p + 2] == 0x0F && code[p + 3] == 0xAE && code[p + 4] == 0xD0) {
            width = 5; // wrfsbase host FS
            push_instruction = false;
        } else {
            return false;
        }
        p += width;
        if (push_instruction) ++pushes;
    }
    return false;
}

std::vector<uint8_t> emit(bool unimpl) {
    std::array<uint8_t, 96> bytes{};
    const size_t size = prosper::emit_guest_fs_swap_stub(bytes.data(), 0x12345678, 0x1122334455667788,
                                                        unimpl);
    return {bytes.begin(), bytes.begin() + size};
}

TEST(GuestFsSwapStub, SavesGuestFsAtExpectedHandlerOffsetAndMaintainsAlignment) {
    for (const bool unimpl : {false, true}) {
        const auto code = emit(unimpl);
        // Optional mov edi, index (5 bytes), movabs r10, fn (10), FS probe/cmp/jne (18).
        const size_t guest_begin = (unimpl ? 5 : 0) + 10 + 5 + 7 + 4 + 2;
        Layout layout;
        ASSERT_TRUE(inspect_guest_call(code, guest_begin, layout)) << "generated guest path decodes";
        EXPECT_TRUE(preserves_swap_stub_contract(layout))
            << "guest FS slot stays at +0x28 and call site retains five-qword alignment";

        // Positive control: an accidental additional push before call shifts the saved FS slot and
        // violates call-site alignment; prove the test's discriminator catches both contracts.
        auto regressed = code;
        constexpr std::array<uint8_t, 3> call_opcode{0x41, 0xFF, 0xD2};
        const auto call = std::search(regressed.begin(), regressed.end(),
                                      call_opcode.begin(), call_opcode.end());
        ASSERT_NE(call, regressed.end()) << "generated stub contains the handler call";
        regressed.insert(call, 0x50); // push rax immediately before call
        Layout broken_layout;
        ASSERT_TRUE(inspect_guest_call(regressed, guest_begin, broken_layout))
            << "mutated guest path still decodes";
        EXPECT_FALSE(preserves_swap_stub_contract(broken_layout))
            << "an added push changes the slot and alignment contract";
    }
}

} // namespace
