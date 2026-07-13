# Shader inspect

`shader_inspect` decodes one raw RDNA2 shader captured with `PROSPER_SHADER_DUMP`. It is the fast
offline path for mapping a failed shader's control flow before changing the recompiler.

```bash
cmake --build build-linux -j8 --target shader_inspect
./build-linux/shader_inspect /tmp/shaders/exec_ps_7f123400.bin
```

Each instruction row includes its dword PC, encoded length, format/opcode, exact raw words, decoded
operands, and, for SOPP branches, the signed immediate and resolved target PC. The header also prints
the stage-agnostic `recompile_coverage` result. That coverage is intentionally compute-safe; a VCC
control-flow shape may remain listed there even when the vertex/fragment structurizer accepts it.

The input is bounded to 16 MiB and decoding stops at the first `s_endpgm` or unknown instruction.
Raw dumps contain title-derived code, are local-only, and must not be committed.
