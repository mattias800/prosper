# Guest/host ABI fixtures

These tests execute project-generated import bridges and compare argument placement with the
compiler's calling conventions. Keep fixed-signature conversion separate from bare guest-ABI
tail transfers: the latter must preserve the complete variadic frame, including AL and overflow
arguments, without a host return hook. Raw register witnesses supplement, not replace, actual
compiler-produced variadic-reader and printf-family fixtures.

The isolated variadic fixture owns only two compiler readers and extern scalar/array references;
callers, golden values, packing and assertions stay on the main compiler. The native-Clang profile
builds those two readers with the same-sysroot GCC without skipping any reference checks.
