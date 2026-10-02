# Calling-convention boundary tests

Actual guest-call placement, stub execution, registry types, variadic frame capture and host CRT
consumption. The isolated variadic fixture owns only two compiler readers and extern scalar/array
references; callers, golden values, packing and assertions stay on the main compiler. The native-Clang
profile builds those two readers with the same-sysroot GCC without skipping any reference checks.
