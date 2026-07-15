# Shader inspect

`shader_inspect` decodes one raw RDNA2 shader. It is the fast offline path for mapping shader control
flow before changing the recompiler.

```bash
cmake --build build-linux -j8 --target shader_inspect
./build-linux/shader_inspect /tmp/shaders/exec_ps_7f123400.bin
```

`PROSPER_SHADER_DUMP=DIR` writes raw shaders that fail recompilation. To inspect a shader that succeeds,
run the title or replay with `PROSPER_SHADER_DUMP_SUCCESS=DIR`. Each unique successful graphics shader
produces a pair such as:

```text
success_ps_35956264829da0c6_62ad8faab4566b7a.bin
success_ps_35956264829da0c6_62ad8faab4566b7a.spv
```

The first hash identifies the translated SPIR-V and the second identifies the exact raw RDNA2 stream.
Pass the `.bin` file to `shader_inspect`; use the adjacent `.spv` for SPIR-V disassembly or validation.
The dump directory is created automatically. `vs` and `ps` in filenames identify vertex and pixel stages.

Each instruction row includes its dword PC, encoded length, format/opcode, exact raw words, decoded
operands, and, for SOPP branches, the signed immediate and resolved target PC. The header also prints
the stage-agnostic `recompile_coverage` result. That coverage is intentionally compute-safe; a VCC
control-flow shape may remain listed there even when the vertex/fragment structurizer accepts it.

The input is bounded to 16 MiB and decoding stops at the first `s_endpgm` or unknown instruction.
Raw and SPIR-V dumps contain title-derived code, are local-only, and must not be committed.
