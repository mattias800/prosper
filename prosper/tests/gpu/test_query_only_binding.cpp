// test_query_only_binding — #657: a descriptor whose EVERY image access is an OpImageQuery* is
// reflected as `query_only`, and one that also reads a texel is not.
//
// Why the flag has to exist at all, rather than reusing `readable`: a query DOES read the
// descriptor, and the reflection deliberately marks the query family readable AND texel_access
// (shader_resources.cpp, the OpImageQueryFormat..Samples arm — its own comment says "Query-only
// descriptors must remain in the reflected layout, and their reported dimensions/LOD contract
// requires the guest's exact extent"). So `readable` cannot separate the two cases; it is true for
// both. What distinguishes them is that a query needs no CONTENT.
//
// That distinction is what lets live compute admit a layered binding it would otherwise decline:
// the shapes it refuses are refused because materialising their texels is unsolved (tiling, layer
// stride, format conversion, upload), and none of that applies when nothing reads a texel. Sonic
// Frontiers' Cyber Space dispatch 0x2005a11600 binds a 64x64x18 cube array — three cubes, declared
// by the module as a 1D image — used only by OpImageQuerySizeLod and OpImageQueryLevels (#2790).
//
// THE NEGATIVE CONTROL IS THE POINT. The positive arm alone passes against an implementation that
// sets query_only unconditionally; the sampling arm is what proves the flag tracks the module's
// actual use. Both modules are identical but for the one sample instruction.
#include "gpu/resources/shader_resources.hpp"

#include <cstdio>
#include <cstdint>
#include <vector>

using namespace prosper::gpu;

namespace {

using Words = std::vector<uint32_t>;

int failures = 0;
void check(bool ok, const char* message) {
    std::printf("[%s] %s\n", ok ? "ok" : "FAIL", message);
    failures += !ok;
}

void put(Words& w, uint32_t op, std::initializer_list<uint32_t> args) {
    w.push_back((uint32_t(args.size() + 1) << 16) | op);
    w.insert(w.end(), args.begin(), args.end());
}

// A minimal compute module binding one sampled 1D image at set 0 / binding 0. `sample` adds a
// single OpImageSampleExplicitLod on the SAME binding; everything else is identical, so the two
// modules differ by exactly the property under test.
Words module_with(bool sample) {
    // ids: 1 void, 2 fn, 3 uint, 4 float, 5 v4float, 10 image, 11 sampled-image,
    //      12 ptr, 13 variable, 14 const, 40 main
    Words w = {0x07230203u, 0x00010300u, 0, 100, 0};
    put(w, 17, {1});                       // OpCapability Shader
    put(w, 17, {43});                      // OpCapability Sampled1D  (a 1D sampled image needs it)
    put(w, 14, {0, 1});                    // OpMemoryModel Logical GLSL450
    put(w, 15, {5, 40, 0x6e69616du, 0});   // OpEntryPoint GLCompute %40 "main"
    put(w, 16, {40, 17, 1, 1, 1});         // OpExecutionMode LocalSize 1 1 1
    put(w, 71, {13, 34, 0});               // OpDecorate %13 DescriptorSet 0
    put(w, 71, {13, 33, 0});               // OpDecorate %13 Binding 0
    put(w, 19, {1});                       // OpTypeVoid
    put(w, 33, {2, 1});                    // OpTypeFunction %void
    put(w, 21, {3, 32, 0});                // OpTypeInt 32 unsigned
    put(w, 22, {4, 32});                   // OpTypeFloat 32
    put(w, 23, {5, 4, 4});                 // OpTypeVector %float 4
    put(w, 25, {10, 4, 0, 0, 0, 0, 1, 0}); // OpTypeImage %float 1D 0 0 0 Sampled=1 Unknown
    put(w, 27, {11, 10});                  // OpTypeSampledImage %10
    put(w, 32, {12, 0, 11});               // OpTypePointer UniformConstant %11
    put(w, 43, {3, 14, 0});                // OpConstant %uint 0
    put(w, 59, {12, 13, 0});               // OpVariable %12 UniformConstant   <- binding 0
    put(w, 54, {1, 40, 0, 2});             // OpFunction %void main
    put(w, 248, {41});                     // OpLabel
    put(w, 61, {11, 42, 13});              // OpLoad %11 %42 %13
    put(w, 100, {10, 43, 42});             // OpImage %10 %43 %42
    put(w, 103, {3, 44, 43, 14});          // OpImageQuerySizeLod %uint %44 %43 %14
    put(w, 104, {3, 45, 43});              // OpImageQueryLevels %uint %45 %43
    if (sample)                            // ...and, for the control, one real texel read:
        put(w, 88, {5, 46, 42, 14, 2, 14});// OpImageSampleExplicitLod %v4float %46 %42 %14 Lod %14
    put(w, 253, {});                       // OpReturn
    put(w, 56, {});                        // OpFunctionEnd
    return w;
}

const SpirvDescriptorBinding* binding_zero(const DescriptorValidationReport& report) {
    for (const auto& d : report.descriptors)
        if (d.set == 0 && d.binding == 0) return &d;
    return nullptr;
}

} // namespace

int main() {
    std::printf("== test_query_only_binding (#657) ==\n");

    const Words query_only_module = module_with(false);
    const Words sampling_module = module_with(true);

    const DescriptorValidationReport query_report = validate_spirv_descriptor_interface(
        query_only_module, nullptr, 0u, SpirvShaderStage::Compute, false);
    const DescriptorValidationReport sample_report = validate_spirv_descriptor_interface(
        sampling_module, nullptr, 0u, SpirvShaderStage::Compute, false);

    const SpirvDescriptorBinding* q = binding_zero(query_report);
    const SpirvDescriptorBinding* s = binding_zero(sample_report);

    check(q != nullptr, "the query-only module reflects a binding at set 0 / binding 0");
    check(s != nullptr, "the sampling module reflects the same binding");
    if (!q || !s) { std::printf("== FAIL: %d ==\n", failures ? failures : 1); return 1; }

    // THE ARM.
    check(q->query_only,
          "a binding used only by OpImageQuerySizeLod/OpImageQueryLevels is query_only");
    // THE CONTROL: one added sample instruction must flip it, and nothing else differs.
    check(!s->query_only,
          "control: adding ONE OpImageSampleExplicitLod on the same binding clears query_only");

    // `readable` cannot stand in for this: it is true in BOTH modules, which is exactly why the
    // flag was added rather than reusing it. If this ever goes red, the reflection changed and the
    // live-compute gate that keys on query_only needs rechecking, not this assertion relaxing.
    check(q->readable && s->readable,
          "both are `readable` -- so `readable` could not have told them apart");

    std::printf(failures ? "== FAIL: %d ==\n" : "== PASS ==\n", failures);
    return failures ? 1 : 0;
}
