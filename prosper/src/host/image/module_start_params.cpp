// module_start_params.cpp — see module_start_params.hpp.

#include "host/image/module_start_params.hpp"

#include "host/image/boot_program.hpp"

namespace prosper {

std::vector<std::pair<uint64_t, uint64_t>> module_start_param_ranges() {
    return {
        // Il2cppUserAssemblies.prx. The Unity PSN package does not always ship as its own PSN.prx:
        // with IL2CPP its native half can be linked straight into the user-assemblies module, and
        // then it is THAT module's module_start which stashes (argc, argp) for the handshake.
        // PGA TOUR 2K25 (PPSA17952) is the measured case — its module_start stores both arguments,
        // and the managed PSN initializer later requires argc & ~0xF and validates
        // argp->{size==0x10, version==0x200}. Started with (0, NULL) it takes the mismatch branch,
        // prints "PSN is an old version that cannot be used by the current player runtime", and
        // then reads argp->version off the NULL argp — SIGSEGV at addr=0x4, 1.2 s into every boot.
        // Same descriptor as PSN.prx below, because it is the same Unity package.
        // CONFIDENCE: HIGH (guest disassembly at Il2cpp+0x6a0b4..0x6a112, one writer found by
        // tools/re/xref.py). The upper bound is BOOT_PSNCORE because that is exactly how
        // boot_program.hpp's own address classifier bounds the Il2cpp image.
        { BOOT_IL2CPP,   BOOT_PSNCORE },
        // Sony's native PSN.prx / SaveData.prx Unity plugins, the original case: their user
        // module_start dereferences argp and null-faults when started with (0, NULL).
        { BOOT_PSN,      BOOT_SAVEDATA },
        { BOOT_SAVEDATA, BOOT_LIBC },
        // PPSA02664's split Unity PSN native plugin (PSNCore.prx + PSNCommon.prx).
        { BOOT_PSNCORE,  0x490000000ull },
        { BOOT_PSNCOMMON, BOOT_COMMONDIALOG },
    };
}

bool module_start_wants_param_descriptor(uint64_t addr) {
    for (const auto& r : module_start_param_ranges())
        if (addr >= r.first && addr < r.second) return true;
    return false;
}

} // namespace prosper
