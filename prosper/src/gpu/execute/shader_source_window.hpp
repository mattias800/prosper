#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace prosper::gpu {
struct AgcShaderHeader;

// Bounds for raw host-side shader walkers, not shader/entry/resource authority. Callers retain
// the normal submitted-input stability contract; this observation does not exclude CPU writers
// or arbitrary native remaps. No code word is read to discover the range.
//
// With a registered header, authenticate its ENTIRE declared byte extent before returning the
// smaller caller budget. Without one, return only the actual contiguous readable dword prefix
// of that budget. A short/headerless shader therefore need not map an artificial maximum blob.
size_t shader_source_dwords(uint64_t address, size_t maximum_dwords,
                            const AgcShaderHeader* registered_header = nullptr);

// Diagnostic/native entry without an already selected header. A present but malformed or
// unreadable registration never falls back to a guessed headerless window.
size_t native_shader_source_dwords(uint64_t address, size_t maximum_dwords);

// Cold walkers only: reject a cut-off encoded instruction rather than laundering the decoder's
// legacy clamped operand storage into a complete native program. Ordinary bounded prefixes need
// not contain END (the scalar fold also accepts such streams); unknown full words stay observable.
// The caller must first bound dwords with shader_source_dwords; these do not authorize guest VA.
size_t shader_source_code_span(const uint32_t* code, size_t dwords);
std::vector<uint32_t> shader_source_snapshot(const uint32_t* code, size_t dwords);
} // namespace prosper::gpu
