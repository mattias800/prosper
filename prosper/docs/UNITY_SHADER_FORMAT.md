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
wrong size. No clean `202012090` magic appears near the crash in a 103KB dump (the one at rel-base `0x2568`
is coincidental — parsing from it yields garbage stats), so the crash is likely in the typetree string
table / param reflection rather than a blob `ShaderSubProgram`, OR the record layout isn't statically
anchorable. **Open:** why the game misparses its own (correctly-decompressed — strings are valid) shader
data. Static byte-analysis has been exhausted; the decisive next step is a LIVE cursor trace of the crash
record's parse — feasible now that single-step works under guest-fs and the deser driver `eboot+0x1612c70`
is hit only 6 times (the 6th is the crash): HWBP that driver, and on hit 6 trace the cursor advances to
find the one field read with the wrong size.
