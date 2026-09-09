"""A correct id-operand model for the SPIR-V opcodes these modules actually contain.

The first closure followed every trailing word as an SSA id. Many SPIR-V instructions carry LITERAL
words there, and a literal that happens to equal a result id creates a dataflow edge that does not
exist. Measured in the dumped modules: 2392 OpExtInst (word 4 is a literal instruction number,
GLSL.std.450 numbers run to ~81 -- squarely in the range of low result ids), 2702 OpConstant (value
words), 1979 OpCompositeExtract (literal indices).

That is exactly how a vote reached an output in the module I traced by hand: the chain began
`id=79 Any -> id=2463 OpExtInst`, and 79 was the *instruction number* of a GLSL op, not the vote.

Returns the word indices that are genuinely ids, per opcode. Unknown opcodes fall back to "all
trailing words are ids", which over-approximates -- the safe direction for a predicate whose negative
answer admits a shader.
"""

# result id lives at word 1 (no type) for these; everything else with a result uses word 2
RESULT_AT_1 = {19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,  # OpType*
               11, 14, 15, 315}                                             # ExtInstImport etc.
NO_RESULT = {62, 71, 72, 249, 250, 246, 247, 248, 252, 253, 254, 5, 6, 7, 8, 10, 17}


def id_operand_indices(op, n):
    """Indices into the instruction words that hold SSA ids (excluding the result)."""
    if op == 12:                       # OpExtInst: type, result, set(id), LITERAL instr, ids...
        return [3] + list(range(5, n))
    if op in (43, 50):                 # OpConstant / OpSpecConstant: value words are literals
        return []
    if op == 81:                       # OpCompositeExtract: composite(id), then literal indices
        return [3] if n > 3 else []
    if op == 82:                       # OpCompositeInsert: object(id), composite(id), literals
        return [i for i in (3, 4) if i < n]
    if op == 79:                       # OpVectorShuffle: v1(id), v2(id), literal components
        return [i for i in (3, 4) if i < n]
    if op in (87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98):
        # OpImageSample*: image(id), coord(id), LITERAL operands mask, then ids
        return [i for i in (3, 4) if i < n] + list(range(6, n))
    if op == 71:                       # OpDecorate: target(id), then literals
        return [1] if n > 1 else []
    if op == 72:                       # OpMemberDecorate: struct(id), literal member, literals
        return [1] if n > 1 else []
    if op in (21, 22):                 # OpTypeInt/Float: result at 1, rest literal
        return []
    if op == 23:                       # OpTypeVector: result at 1, component type(id), literal count
        return [2] if n > 2 else []
    if op == 24:                       # OpTypeMatrix: column type(id), literal count
        return [2] if n > 2 else []
    if op == 25:                       # OpTypeImage: sampled type(id), then literals
        return [2] if n > 2 else []
    if op == 28:                       # OpTypeArray: element type(id), length(id) -- both ids
        return [i for i in (2, 3) if i < n]
    if op == 32:                       # OpTypePointer: result at 1, LITERAL storage, type(id)
        return [3] if n > 3 else []
    if op == 59:                       # OpVariable: type, result, LITERAL storage, [initializer]
        return [4] if n > 4 else []
    if op == 33:                       # OpTypeFunction: return type + param types, all ids
        return list(range(2, n))
    if op in NO_RESULT:
        return list(range(1, n))
    return list(range(3, n))           # default: type, result, then ids


def result_index(op):
    if op in NO_RESULT:
        return None
    return 1 if op in RESULT_AT_1 else 2
