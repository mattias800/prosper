// Offline SPI_PS_INPUT_CNTL linkage classification (see tools/gpu_replay/pixel_input_linkage.hpp).
//
// The distinction this pins down is behavioral, not cosmetic: an input wired to the constant form
// (OFFSET=0x20) consumes NO producer export, so reporting it as "param 32" — or worse, silently as
// "param 0" after masking — would tell an investigator that a varying carries real vertex-stage
// data when the interpolator is actually synthesizing a constant. That is precisely the question an
// offline capsule investigation needs answered, so every branch is asserted here.

#include "../tools/gpu_replay/pixel_input_linkage.hpp"

#include <cstdio>

using namespace prosper::gpu;
using namespace prosper::gpu::replay_tool;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { std::printf("  [FAIL] %s\n", m); ++fails; } \
                         else std::printf("  [ok]   %s\n", m); } while (0)

int main() {
    std::printf("== test_gpu_replay_pixel_inputs ==\n");

    {
        // A slot outside valid_mask consumes nothing, whatever its stale control word says.
        PixelInputMapping mapping;
        mapping.controls[4] = 7;
        mapping.valid_mask = 0;
        const auto linkage = pixel_input_linkage(mapping, 4);
        CHECK(linkage.kind == PixelInputKind::Unused, "invalid slot reports Unused");
    }

    {
        // Ordinary linkage: OFFSET names the producer PARAM slot directly.
        PixelInputMapping mapping;
        mapping.controls[1] = 1;
        mapping.controls[2] = 3;
        mapping.valid_mask = (1u << 1) | (1u << 2);
        const auto one = pixel_input_linkage(mapping, 1);
        CHECK(one.kind == PixelInputKind::Param && one.param == 1, "input1 -> PARAM1");
        const auto two = pixel_input_linkage(mapping, 2);
        CHECK(two.kind == PixelInputKind::Param && two.param == 3, "input2 -> PARAM3");
    }

    {
        // OFFSET=0x20 is the constant form. This is the case the whole diagnostic exists for:
        // it must NOT be reported as consuming a PARAM export.
        PixelInputMapping mapping;
        mapping.controls[6] = 0x20u;
        mapping.valid_mask = 1u << 6;
        const auto linkage = pixel_input_linkage(mapping, 6);
        CHECK(linkage.kind == PixelInputKind::ConstantDefault,
              "OFFSET=0x20 reports ConstantDefault, not a PARAM consumer");
        CHECK(!linkage.default_xyz_one && !linkage.default_w_one,
              "DEFAULT_VAL=0 -> xyz 0.0, w 0.0");
    }

    {
        // GFX10 DEFAULT_VAL picks 0.0/1.0 independently for xyz and for w. A varying defaulted to
        // 1.0 is indistinguishable from a legitimately-computed 1.0 downstream, so decode it here.
        PixelInputMapping mapping;
        mapping.valid_mask = (1u << 0) | (1u << 1) | (1u << 2);
        mapping.controls[0] = 0x20u | (1u << 8);   // DEFAULT_VAL=1
        mapping.controls[1] = 0x20u | (2u << 8);   // DEFAULT_VAL=2
        mapping.controls[2] = 0x20u | (3u << 8);   // DEFAULT_VAL=3
        const auto w_one = pixel_input_linkage(mapping, 0);
        CHECK(w_one.kind == PixelInputKind::ConstantDefault &&
              !w_one.default_xyz_one && w_one.default_w_one,
              "DEFAULT_VAL=1 -> xyz 0.0, w 1.0");
        const auto xyz_one = pixel_input_linkage(mapping, 1);
        CHECK(xyz_one.kind == PixelInputKind::ConstantDefault &&
              xyz_one.default_xyz_one && !xyz_one.default_w_one,
              "DEFAULT_VAL=2 -> xyz 1.0, w 0.0");
        const auto both = pixel_input_linkage(mapping, 2);
        CHECK(both.kind == PixelInputKind::ConstantDefault &&
              both.default_xyz_one && both.default_w_one,
              "DEFAULT_VAL=3 -> xyz 1.0, w 1.0");
    }

    {
        // Explicit parameter-cache pass-through: OFFSET bit 5 + FLAT_SHADE with a nonzero low
        // OFFSET. The consumed PARAM is the LOW five bits, not the full six-bit OFFSET field.
        PixelInputMapping mapping;
        mapping.controls[3] = 0x420u | 5u;
        mapping.valid_mask = 1u << 3;
        const auto linkage = pixel_input_linkage(mapping, 3);
        CHECK(linkage.kind == PixelInputKind::ParamPassthrough && linkage.param == 5,
              "pass-through encoding -> PARAM5 via the low five OFFSET bits");
    }

    {
        // The ambiguous PARAM0 pass-through encoding (0x420 with a zero low OFFSET) is NOT treated
        // as pass-through by effective_passthrough_mask(), so it falls through to the constant
        // form. Pinned so the classifier keeps agreeing with the recompiler's own predicate.
        PixelInputMapping mapping;
        mapping.controls[3] = 0x420u;
        mapping.valid_mask = 1u << 3;
        const auto linkage = pixel_input_linkage(mapping, 3);
        CHECK(linkage.kind == PixelInputKind::ConstantDefault,
              "ambiguous 0x420 PARAM0 encoding follows the constant form");
    }

    {
        // An explicitly-flagged pass-through slot is honored even without the register encoding
        // (this is the metadata-derived path).
        PixelInputMapping mapping;
        mapping.controls[7] = 2u;
        mapping.valid_mask = 1u << 7;
        mapping.passthrough_mask = 1u << 7;
        const auto linkage = pixel_input_linkage(mapping, 7);
        CHECK(linkage.kind == PixelInputKind::ParamPassthrough && linkage.param == 2,
              "explicit passthrough_mask slot -> ParamPassthrough");
    }

    {
        // Out-of-range slots never read past the fixed 32-entry control array.
        PixelInputMapping mapping;
        mapping.valid_mask = 0xffffffffu;
        const auto linkage = pixel_input_linkage(mapping, 32);
        CHECK(linkage.kind == PixelInputKind::Unused, "slot 32 is out of range and reports Unused");
    }

    std::printf(fails ? "== FAILED (%d) ==\n" : "== PASSED ==\n", fails);
    return fails ? 1 : 0;
}
