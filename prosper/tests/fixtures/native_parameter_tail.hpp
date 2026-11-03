// Native interpolation plus an original saved-mask/SCC0 trailer beyond the first END.
#pragma once
#include "fixtures/native_pixel_parameter.hpp"

namespace prosper::test::native_parameter_tail {
using Words = native_pixel_parameter::Words;
inline Words discard() {
    // LLVM gfx1030: EXEC=0; EXP MRT0 EN0 DONE; END. No output payload is read.
    return {0xbefe0480u, 0xf8000800u, 0x00000000u, 0xbf810000u};
}
inline Words program(bool smooth = false) {
    auto code = native_pixel_parameter::program(30, smooth);
    // LLVM gfx1030: save EXEC; CMP 0.5<v8; saved &= VCC; SCC0 skips EXP/END; EXEC=saved.
    const Words branch{0xbe84047eu, 0x7c0210f0u, 0x87846a04u, 0xbf840004u, 0xbefe0404u};
    code.insert(code.end() - 3, branch.begin(), branch.end());
    const auto tail = discard();
    code.insert(code.end(), tail.begin(), tail.end());
    return code;
}
} // namespace prosper::test::native_parameter_tail
