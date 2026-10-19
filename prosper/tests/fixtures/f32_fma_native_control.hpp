// Independent GLSL Fma control shared by the standard and mixed F32 execution witnesses.
#pragma once
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <cstdint>
#include <initializer_list>
#include <vector>
namespace prosper::test::fma_rounding {
using namespace prosper::gpu;
// Independent, hand-built SPIR-V: buffer3 contains A, B, C and result arrays of 64
// raw words. Dynamic storage loads prevent constant folding of the accuracy probe.
// Do not use SpirvCompute::fma here: a wrong guest emission must not change its oracle.
inline std::vector<uint32_t> native_fma_control() {
    std::vector<uint32_t> module{0x07230203, 0x00010300, 0, 35, 0};
    const auto emit = [&](uint32_t opcode, std::initializer_list<uint32_t> operands) {
        module.push_back((uint32_t(operands.size() + 1) << 16) | opcode);
        module.insert(module.end(), operands);
    };
    emit(Op_Capability, {1}); // Shader
    emit(Op_ExtInstImport, {1, 0x4c534c47, 0x6474732e, 0x3035342e, 0}); // GLSL.std.450
    emit(Op_MemoryModel, {0, 1}); // Logical, GLSL450
    emit(Op_EntryPoint, {5, 16, 0x6e69616d, 0, 14}); // GLCompute main, GlobalInvocationId
    emit(Op_ExecutionMode, {16, 17, 64, 1, 1}); // LocalSize
    emit(Op_Decorate, {8, 6, 4}); // ArrayStride
    emit(Op_Decorate, {9, 2}); // Block
    emit(Op_MemberDecorate, {9, 0, 35, 0}); // Offset
    emit(Op_Decorate, {15, 34, 0}); // DescriptorSet
    emit(Op_Decorate, {15, 33, 3}); // Binding
    emit(Op_Decorate, {14, 11, 28}); // BuiltIn GlobalInvocationId
    emit(Op_TypeVoid, {2});
    emit(Op_TypeFunction, {3, 2});
    emit(Op_TypeInt, {4, 32, 0});
    emit(Op_TypeFloat, {5, 32});
    emit(Op_TypeVector, {6, 4, 3});
    emit(Op_TypePointer, {7, 1, 6}); // Input uvec3
    emit(Op_TypeRuntimeArray, {8, 4});
    emit(Op_TypeStruct, {9, 8});
    emit(Op_TypePointer, {10, 12, 9}); // StorageBuffer block
    emit(Op_TypePointer, {11, 12, 4}); // StorageBuffer uint
    emit(Op_Constant, {4, 12, 0});
    emit(Op_Constant, {4, 13, 64});
    emit(Op_Variable, {7, 14, 1});
    emit(Op_Variable, {10, 15, 12});
    emit(Op_Function, {2, 16, 0, 3});
    emit(Op_Label, {17});
    emit(Op_Load, {6, 18, 14});
    emit(Op_CompositeExtract, {4, 19, 18, 0});
    emit(Op_IAdd, {4, 20, 19, 13});
    emit(Op_IAdd, {4, 21, 20, 13});
    emit(Op_IAdd, {4, 22, 21, 13});
    for (uint32_t k = 0; k < 3; ++k) {
        emit(Op_AccessChain, {11, 23 + k, 15, 12, 19 + k});
        emit(Op_Load, {4, 26 + k, 23 + k});
        emit(Op_Bitcast, {5, 29 + k, 26 + k});
    }
    emit(Op_ExtInst, {5, 32, 1, 50, 29, 30, 31}); // GLSL.std.450 Fma, directly from SPIR-V
    emit(Op_Bitcast, {4, 33, 32});
    emit(Op_AccessChain, {11, 34, 15, 12, 22});
    emit(Op_Store, {34, 33});
    emit(Op_Return, {});
    emit(Op_FunctionEnd, {});
    return module;
}

}   // namespace prosper::test::fma_rounding
