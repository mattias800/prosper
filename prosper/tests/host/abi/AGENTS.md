# Guest/host ABI fixtures

These tests execute project-generated import bridges and compare argument placement with the
compiler's calling conventions. Keep fixed-signature conversion separate from bare guest-ABI
tail transfers: the latter must preserve the complete variadic frame, including AL and overflow
arguments, without a host return hook. Raw register witnesses supplement, not replace, actual
compiler-produced variadic-reader and printf-family fixtures.
