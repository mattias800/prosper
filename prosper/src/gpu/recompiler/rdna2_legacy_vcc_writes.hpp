// Shared legacy implicit VALU VCC destinations; no ADR 0038 scalar-result facts consumed.
#pragma once
namespace prosper::gpu {
struct Rdna2Inst;
bool legacy_valu_writes_vcc(const Rdna2Inst&);
} // namespace prosper::gpu
