# Unity 2022.3.32f1 shader serialization format (for the deser-fault RE)

Verified format from a multi-agent collaboration (Fable, cross-checked against AssetRipper 0.3.4.0
`ShaderBlob/*` + TypeTreeDumps `2022.3.32f1.dump`). This is the reference for the render-loop
deserialization fault (`docs/RENDER_LOOP.md`) — the game crashes deserializing the built-in shader
`Hidden/CubeBlur` on the `-force-gfx-direct`+`PROSPER_GUEST_FS` path.

## Two layers, both present in the ~37KB reader buffer

1. **Typetree `m_ParsedForm` (SerializedShader)** — the on-disk structure. Param reflection here uses
   **16-byte fixed records** with an `m_NameIndex` (s32) into a shared string table; string *values*
   (m_Name=`Hidden/CubeBlur`, tags `RenderType`/`Opaque`) are inline.
2. **Decompressed `m_CompressedBlob`** — Unity's shader-program blob. Per `platforms[i]`, the slice
   `compressedBlob[offsets[i]..+compressedLengths[i]]` is **LZ4-block** decompressed to
   `decompressedLengths[i]` bytes. Segment 0 begins with an entry table:
   `s32 entryCount; entryCount × { s32 offset; s32 length; s32 segment }`. Each entry at `offset` is a
   `ShaderSubProgram` record where **all reflection names are inline length-prefixed 4-aligned strings**.

`alignedString` = `{u32 len}{len bytes, no NUL}{pad to 4}` (len==0 → just the u32). This is the reader
`eboot+0x7e4090` in prosper's trace; the array driver is `eboot+0x7fc9f0` ({s32 count}{count × alignedString}).

## `ShaderSubProgram` record (2022.3, version magic 202012090)

```
s32 version              // 202012090 for all of 2021.2+
s32 programType          // ShaderGpuProgramType
s32 statsALU; s32 statsTEX; s32 statsFlow; s32 statsTempRegister
s32 keywordCount; keywordCount × alignedString      // MERGED keyword list (2021.2+; no local/global split)
s32 programDataLen; bytes[programDataLen]; align4    // the platform (PS5/AGC) shader binary
s32 sourceMap
s32 bindCount; bindCount × { u32 source; u32 target }
s32 paramGroupCount      // group 0 = loose/$Globals; groups 1..N = real cbuffers (UnityPerDraw, …)
  per group:
    alignedString name          // e.g. "UnityPerDraw"
    s32 usedSize                // cbuffer size IN BYTES (comes right after the name)
    s32 paramCount
    paramCount × MEMBER:
        alignedString paramName // "unity_ObjectToWorld", …
        s32 type                // ShaderParamType 0=Float 1=Int 2=Bool 3=Half 4=Short 5=UInt
        s32 rowCount
        s32 columnCount         // "Dim" for vectors
        s32 isMatrix            // full s32, 0/1
        s32 arraySize
        s32 index               // byte offset within the cbuffer
    s32 structCount             // 2017.3+; usually 0
    structCount × { alignedString name; s32 index; s32 arraySize; s32 structSize;
                    s32 memberCount; memberCount × MEMBER }
s32 paramGroup2Count     // textures/samplers/buffers/UAVs/cbuffer BINDINGS
  per entry:
    alignedString name
    s32 type                    // 0=texture 1=cbuffer-binding 2=buffer 3=UAV 4=sampler
    s32 index                   // bind slot
    s32 extraValue
    if type==0: u32 texExtra    // 2018.2+: bit0=multisampled, bits1..=dimension; extraValue=samplerIndex
```

**Each cbuffer member = aligned name + 24 bytes (6× s32), variable stride** (NOT a 16-byte fixed record).
Note: a cbuffer name appears TWICE — as a param-group header (reflection) and again in group2 as a
`type==1` binding entry. `usedSize` for UnityPerDraw is a plausible ~0x230.

Version diff 2021.2→2022.3 (typetree `SerializedProgram`): 2022.3 ADDED `m_PlayerSubPrograms`
(`vector<vector<SerializedPlayerSubProgram>>`), `m_ParameterBlobIndices` (`vector<vector<u32>>`), and
`m_SerializedKeywordStateMask` (`vector<u16>`, align after), plus `stageCounts` after `compressedBlob`.
`SerializedSubProgram.m_ShaderRequirements` is s64. For blob versions `201806140 ≤ v < 202012090` there is
an EXTRA local-keywords aligned-string array after the global keywords (ABSENT in 2022.3) — a mis-branch
here would read a phantom array out of the bytecode.

## The fault (as of this session)

The reader reads a garbage `{count}` (`0x400` at buffer offset `0x7a20`) for an inline-string array, loops
reading "strings" out of the shader bytecode, and reads a garbage length at `0x7f70` → `alloc(~16 EiB)` →
null → `movb $0,[0]` crash at `eboot+0x46beb4`. The reader's bounds check (`cursor+4 ≤ end`) PASSES, so the
`end` is correct — the CURSOR is mis-positioned within valid bounds from an upstream field read with the
wrong size.

### Confirmed this session (Fable collaboration + static walk of the 103KB reader-buffer dump)

- **Layer = typetree `m_Parameters`, NOT the LZ4 blob.** No `202012090` magic near the crash (the one at
  rel-base `0x2568` is coincidental). The bytes decode cleanly as Fable's typetree `SerializedProgramParameters`:
  16-byte `VectorParameter`/`MatrixParameter` records (`{s32 nameIdx; s32 index; s32 arraySize; s8 type; s8 dim; 2pad}`)
  and `ConstantBuffer` records. Verified anchor at buffer off `0x7080`: a clean CB — `nameIdx=5`, mat 0,
  **vec 3** (three floats, index 0/4/8, type=0 Float dim=1), struct 0, `m_Size=0x10`, `isPartialCB=1`.
- **`0x400` is NOT a cbuffer `m_Size`.** Forward-parsing the CB array from `0x7080` shows the real cbuffers
  are tiny; by the 8th "CB" the parse walks into the `<none>` string bytes. So Fable's MED-confidence
  `0x400=UnityStereoGlobals.m_Size` did not hold — the `0x7a20` divergence is in a LATER subprogram in the
  `m_SubPrograms`/`m_PlayerSubPrograms` stream, reached after an upstream field-size/align desync.
- **prosper has NO Unity shader-reflection/typetree parser** (only `gpu/agc_shader_layout.cpp` = RDNA2 GPU
  resource binding, unrelated). So the crashing reader is **100% game-side Unity 2022.3 code** on a buffer
  prosper does not produce. ⇒ the bug is EITHER (a) data our env emitted diverges from 2022.3 layout, OR
  (b) a value we fed upstream flipped the parse dispatch. CubeBlur is the FIRST built-in with populated
  stereo variants (`m_PlayerSubPrograms` non-empty, odd-length `m_KeywordIndices` needing align-pad,
  `UnityStereoGlobals`); a reader with a 2021.2-shaped model silently survives every earlier shader (empty
  fields read as 0) and dies only here.

### Fable's three ranked off-by-one candidates (deficit arithmetic distinguishes them)

1. `m_ShaderRequirements` consumed as **s32 (4B) instead of s64 (8B)** — 2022.x widened it. Deficit = 4 ×
   subprograms parsed. **Top suspect.**
2. Missing **align-to-4 after the odd-count `u16[]`** arrays (`m_KeywordIndices`/`m_Channels`). Deficit = 2
   per odd-length array (CubeBlur's per-variant keyword arrays are count=1 = odd; most shaders are 0/even).
3. Missing **`m_IsPartialCB`+3pad** per ConstantBuffer (2021.1+). Deficit = 4 per cbuffer.

### Decisive next step — LIVE cursor trace

Static byte-analysis is exhausted (the data itself parses clean to 2022.3; the desync is dynamic/cumulative).
HWBP the deser driver `eboot+0x1612c70` (hit only ~6 times; the last is the crash), and on the crash-shader
invocation trace the cursor advances field-by-field. At the divergence cursor, per Fable: confirm 16-byte
stride with byte12∈{0..5}/byte13∈{1..4}, walk back to the enclosing `SerializedSubProgram` start, and check
in order (a) requirements consumed as 8B, (b) odd u16[] arrays 4-aligned, (c) `m_IsPartialCB`+pad consumed.
The cursor-deficit arithmetic (4×subprograms vs 2×odd-arrays vs 4×cbuffers) names the single mis-sized field.
