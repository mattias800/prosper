"""Independent RDNA2 instruction decoder and opcode tables.

Written from the AMD "RDNA 2" Instruction Set Architecture Reference Guide (doc 70648),
chapter 13 "Microcode Formats" (field layouts and the per-format opcode tables 64, 66, 68,
70, 72, 74, 76, 78, 81, 83 and the VOP3B opcode table). Nothing here is derived from
prosper's recompiler or from any other emulator. Opcode tables are literal transcriptions
of the ISA tables; LLVM (llvm_check.py) is a third, independent cross-check, and a
disagreement is settled by the ISA document, not by either side.
"""

import struct
from dataclasses import dataclass, field

# Opcode tables: {opcode number: ISA mnemonic}. SOP2 = Table 64, SOPK = 66, SOP1 = 68,
# SOPC = 70, SOPP = 72, SMEM = 74, VOP2 = 76, VOP1 = 78, VOPC = 81, VOP3 = 83 (+ VOP3B).
SOP2 = {
    0: "S_ADD_U32",
    1: "S_SUB_U32",
    2: "S_ADD_I32",
    3: "S_SUB_I32",
    4: "S_ADDC_U32",
    5: "S_SUBB_U32",
    6: "S_MIN_I32",
    7: "S_MIN_U32",
    8: "S_MAX_I32",
    9: "S_MAX_U32",
    10: "S_CSELECT_B32",
    11: "S_CSELECT_B64",
    14: "S_AND_B32",
    15: "S_AND_B64",
    16: "S_OR_B32",
    17: "S_OR_B64",
    18: "S_XOR_B32",
    19: "S_XOR_B64",
    20: "S_ANDN2_B32",
    21: "S_ANDN2_B64",
    22: "S_ORN2_B32",
    23: "S_ORN2_B64",
    24: "S_NAND_B32",
    25: "S_NAND_B64",
    26: "S_NOR_B32",
    27: "S_NOR_B64",
    28: "S_XNOR_B32",
    29: "S_XNOR_B64",
    30: "S_LSHL_B32",
    31: "S_LSHL_B64",
    32: "S_LSHR_B32",
    33: "S_LSHR_B64",
    34: "S_ASHR_I32",
    35: "S_ASHR_I64",
    36: "S_BFM_B32",
    37: "S_BFM_B64",
    38: "S_MUL_I32",
    39: "S_BFE_U32",
    40: "S_BFE_I32",
    41: "S_BFE_U64",
    42: "S_BFE_I64",
    44: "S_ABSDIFF_I32",
    46: "S_LSHL1_ADD_U32",
    47: "S_LSHL2_ADD_U32",
    48: "S_LSHL3_ADD_U32",
    49: "S_LSHL4_ADD_U32",
    50: "S_PACK_LL_B32_B16",
    51: "S_PACK_LH_B32_B16",
    52: "S_PACK_HH_B32_B16",
    53: "S_MUL_HI_U32",
    54: "S_MUL_HI_I32",
}
SOPK = {
    0: "S_MOVK_I32",
    1: "S_VERSION",
    2: "S_CMOVK_I32",
    3: "S_CMPK_EQ_I32",
    4: "S_CMPK_LG_I32",
    5: "S_CMPK_GT_I32",
    6: "S_CMPK_GE_I32",
    7: "S_CMPK_LT_I32",
    8: "S_CMPK_LE_I32",
    9: "S_CMPK_EQ_U32",
    10: "S_CMPK_LG_U32",
    11: "S_CMPK_GT_U32",
    12: "S_CMPK_GE_U32",
    13: "S_CMPK_LT_U32",
    14: "S_CMPK_LE_U32",
    15: "S_ADDK_I32",
    16: "S_MULK_I32",
    18: "S_GETREG_B32",
    19: "S_SETREG_B32",
    21: "S_SETREG_IMM32_B32",
    22: "S_CALL_B64",
    23: "S_WAITCNT_VSCNT",
    24: "S_WAITCNT_VMCNT",
    25: "S_WAITCNT_EXPCNT",
    26: "S_WAITCNT_LGKMCNT",
    27: "S_SUBVECTOR_LOOP_BEGIN",
    28: "S_SUBVECTOR_LOOP_END",
}
SOP1 = {
    3: "S_MOV_B32",
    4: "S_MOV_B64",
    5: "S_CMOV_B32",
    6: "S_CMOV_B64",
    7: "S_NOT_B32",
    8: "S_NOT_B64",
    9: "S_WQM_B32",
    10: "S_WQM_B64",
    11: "S_BREV_B32",
    12: "S_BREV_B64",
    13: "S_BCNT0_I32_B32",
    14: "S_BCNT0_I32_B64",
    15: "S_BCNT1_I32_B32",
    16: "S_BCNT1_I32_B64",
    17: "S_FF0_I32_B32",
    18: "S_FF0_I32_B64",
    19: "S_FF1_I32_B32",
    20: "S_FF1_I32_B64",
    21: "S_FLBIT_I32_B32",
    22: "S_FLBIT_I32_B64",
    23: "S_FLBIT_I32",
    24: "S_FLBIT_I32_I64",
    25: "S_SEXT_I32_I8",
    26: "S_SEXT_I32_I16",
    27: "S_BITSET0_B32",
    28: "S_BITSET0_B64",
    29: "S_BITSET1_B32",
    30: "S_BITSET1_B64",
    31: "S_GETPC_B64",
    32: "S_SETPC_B64",
    33: "S_SWAPPC_B64",
    34: "S_RFE_B64",
    36: "S_AND_SAVEEXEC_B64",
    37: "S_OR_SAVEEXEC_B64",
    38: "S_XOR_SAVEEXEC_B64",
    39: "S_ANDN2_SAVEEXEC_B64",
    40: "S_ORN2_SAVEEXEC_B64",
    41: "S_NAND_SAVEEXEC_B64",
    42: "S_NOR_SAVEEXEC_B64",
    43: "S_XNOR_SAVEEXEC_B64",
    44: "S_QUADMASK_B32",
    45: "S_QUADMASK_B64",
    46: "S_MOVRELS_B32",
    47: "S_MOVRELS_B64",
    48: "S_MOVRELD_B32",
    49: "S_MOVRELD_B64",
    52: "S_ABS_I32",
    55: "S_ANDN1_SAVEEXEC_B64",
    56: "S_ORN1_SAVEEXEC_B64",
    57: "S_ANDN1_WREXEC_B64",
    58: "S_ANDN2_WREXEC_B64",
    59: "S_BITREPLICATE_B64_B32",
    60: "S_AND_SAVEEXEC_B32",
    61: "S_OR_SAVEEXEC_B32",
    62: "S_XOR_SAVEEXEC_B32",
    63: "S_ANDN2_SAVEEXEC_B32",
    64: "S_ORN2_SAVEEXEC_B32",
    65: "S_NAND_SAVEEXEC_B32",
    66: "S_NOR_SAVEEXEC_B32",
    67: "S_XNOR_SAVEEXEC_B32",
    68: "S_ANDN1_SAVEEXEC_B32",
    69: "S_ORN1_SAVEEXEC_B32",
    70: "S_ANDN1_WREXEC_B32",
    71: "S_ANDN2_WREXEC_B32",
    73: "S_MOVRELSD_2_B32",
}
SOPC = {
    0: "S_CMP_EQ_I32",
    1: "S_CMP_LG_I32",
    2: "S_CMP_GT_I32",
    3: "S_CMP_GE_I32",
    4: "S_CMP_LT_I32",
    5: "S_CMP_LE_I32",
    6: "S_CMP_EQ_U32",
    7: "S_CMP_LG_U32",
    8: "S_CMP_GT_U32",
    9: "S_CMP_GE_U32",
    10: "S_CMP_LT_U32",
    11: "S_CMP_LE_U32",
    12: "S_BITCMP0_B32",
    13: "S_BITCMP1_B32",
    14: "S_BITCMP0_B64",
    15: "S_BITCMP1_B64",
    18: "S_CMP_EQ_U64",
    19: "S_CMP_LG_U64",
}
SOPP = {
    0: "S_NOP",
    1: "S_ENDPGM",
    2: "S_BRANCH",
    3: "S_WAKEUP",
    4: "S_CBRANCH_SCC0",
    5: "S_CBRANCH_SCC1",
    6: "S_CBRANCH_VCCZ",
    7: "S_CBRANCH_VCCNZ",
    8: "S_CBRANCH_EXECZ",
    9: "S_CBRANCH_EXECNZ",
    10: "S_BARRIER",
    11: "S_SETKILL",
    12: "S_WAITCNT",
    13: "S_SETHALT",
    14: "S_SLEEP",
    15: "S_SETPRIO",
    16: "S_SENDMSG",
    17: "S_SENDMSGHALT",
    18: "S_TRAP",
    19: "S_ICACHE_INV",
    20: "S_INCPERFLEVEL",
    21: "S_DECPERFLEVEL",
    22: "S_TTRACEDATA",
    23: "S_CBRANCH_CDBGSYS",
    24: "S_CBRANCH_CDBGUSER",
    25: "S_CBRANCH_CDBGSYS_OR_USER",
    26: "S_CBRANCH_CDBGSYS_AND_USER",
    27: "S_ENDPGM_SAVED",
    30: "S_ENDPGM_ORDERED_PS_DONE",
    31: "S_CODE_END",
    32: "S_INST_PREFETCH",
    33: "S_CLAUSE",
    35: "S_WAITCNT_DEPCTR",
    36: "S_ROUND_MODE",
    37: "S_DENORM_MODE",
    40: "S_TTRACEDATA_IMM",
}
SMEM = {
    0: "S_LOAD_DWORD",
    1: "S_LOAD_DWORDX2",
    2: "S_LOAD_DWORDX4",
    3: "S_LOAD_DWORDX8",
    4: "S_LOAD_DWORDX16",
    8: "S_BUFFER_LOAD_DWORD",
    9: "S_BUFFER_LOAD_DWORDX2",
    10: "S_BUFFER_LOAD_DWORDX4",
    11: "S_BUFFER_LOAD_DWORDX8",
    12: "S_BUFFER_LOAD_DWORDX16",
    31: "S_GL1_INV",
    32: "S_DCACHE_INV",
    36: "S_MEMTIME",
    37: "S_MEMREALTIME",
    38: "S_ATC_PROBE",
    39: "S_ATC_PROBE_BUFFER",
}
VOP2 = {
    1: "V_CNDMASK_B32",
    2: "V_DOT2C_F32_F16",
    3: "V_ADD_F32",
    4: "V_SUB_F32",
    5: "V_SUBREV_F32",
    6: "V_FMAC_LEGACY_F32",
    7: "V_MUL_LEGACY_F32",
    8: "V_MUL_F32",
    9: "V_MUL_I32_I24",
    10: "V_MUL_HI_I32_I24",
    11: "V_MUL_U32_U24",
    12: "V_MUL_HI_U32_U24",
    13: "V_DOT4C_I32_I8",
    15: "V_MIN_F32",
    16: "V_MAX_F32",
    17: "V_MIN_I32",
    18: "V_MAX_I32",
    19: "V_MIN_U32",
    20: "V_MAX_U32",
    22: "V_LSHRREV_B32",
    24: "V_ASHRREV_I32",
    26: "V_LSHLREV_B32",
    27: "V_AND_B32",
    28: "V_OR_B32",
    29: "V_XOR_B32",
    30: "V_XNOR_B32",
    37: "V_ADD_NC_U32",
    38: "V_SUB_NC_U32",
    39: "V_SUBREV_NC_U32",
    40: "V_ADD_CO_CI_U32",
    41: "V_SUB_CO_CI_U32",
    42: "V_SUBREV_CO_CI_U32",
    43: "V_FMAC_F32",
    44: "V_FMAMK_F32",
    45: "V_FMAAK_F32",
    47: "V_CVT_PKRTZ_F16_F32",
    50: "V_ADD_F16",
    51: "V_SUB_F16",
    52: "V_SUBREV_F16",
    53: "V_MUL_F16",
    54: "V_FMAC_F16",
    55: "V_FMAMK_F16",
    56: "V_FMAAK_F16",
    57: "V_MAX_F16",
    58: "V_MIN_F16",
    59: "V_LDEXP_F16",
    60: "V_PK_FMAC_F16",
}
VOP1 = {
    0: "V_NOP",
    1: "V_MOV_B32",
    2: "V_READFIRSTLANE_B32",
    3: "V_CVT_I32_F64",
    4: "V_CVT_F64_I32",
    5: "V_CVT_F32_I32",
    6: "V_CVT_F32_U32",
    7: "V_CVT_U32_F32",
    8: "V_CVT_I32_F32",
    10: "V_CVT_F16_F32",
    11: "V_CVT_F32_F16",
    12: "V_CVT_RPI_I32_F32",
    13: "V_CVT_FLR_I32_F32",
    14: "V_CVT_OFF_F32_I4",
    15: "V_CVT_F32_F64",
    16: "V_CVT_F64_F32",
    17: "V_CVT_F32_UBYTE0",
    18: "V_CVT_F32_UBYTE1",
    19: "V_CVT_F32_UBYTE2",
    20: "V_CVT_F32_UBYTE3",
    21: "V_CVT_U32_F64",
    22: "V_CVT_F64_U32",
    23: "V_TRUNC_F64",
    24: "V_CEIL_F64",
    25: "V_RNDNE_F64",
    26: "V_FLOOR_F64",
    27: "V_PIPEFLUSH",
    32: "V_FRACT_F32",
    33: "V_TRUNC_F32",
    34: "V_CEIL_F32",
    35: "V_RNDNE_F32",
    36: "V_FLOOR_F32",
    37: "V_EXP_F32",
    39: "V_LOG_F32",
    42: "V_RCP_F32",
    43: "V_RCP_IFLAG_F32",
    46: "V_RSQ_F32",
    47: "V_RCP_F64",
    49: "V_RSQ_F64",
    51: "V_SQRT_F32",
    52: "V_SQRT_F64",
    53: "V_SIN_F32",
    54: "V_COS_F32",
    55: "V_NOT_B32",
    56: "V_BFREV_B32",
    57: "V_FFBH_U32",
    58: "V_FFBL_B32",
    59: "V_FFBH_I32",
    60: "V_FREXP_EXP_I32_F64",
    61: "V_FREXP_MANT_F64",
    62: "V_FRACT_F64",
    63: "V_FREXP_EXP_I32_F32",
    64: "V_FREXP_MANT_F32",
    65: "V_CLREXCP",
    66: "V_MOVRELD_B32",
    67: "V_MOVRELS_B32",
    68: "V_MOVRELSD_B32",
    72: "V_MOVRELSD_2_B32",
    80: "V_CVT_F16_U16",
    81: "V_CVT_F16_I16",
    82: "V_CVT_U16_F16",
    83: "V_CVT_I16_F16",
    84: "V_RCP_F16",
    85: "V_SQRT_F16",
    86: "V_RSQ_F16",
    87: "V_LOG_F16",
    88: "V_EXP_F16",
    89: "V_FREXP_MANT_F16",
    90: "V_FREXP_EXP_I16_F16",
    91: "V_FLOOR_F16",
    92: "V_CEIL_F16",
    93: "V_TRUNC_F16",
    94: "V_RNDNE_F16",
    95: "V_FRACT_F16",
    96: "V_SIN_F16",
    97: "V_COS_F16",
    98: "V_SAT_PK_U8_I16",
    99: "V_CVT_NORM_I16_F16",
    100: "V_CVT_NORM_U16_F16",
    101: "V_SWAP_B32",
    104: "V_SWAPREL_B32",
}
VOPC = {
    0: "V_CMP_F_F32",
    1: "V_CMP_LT_F32",
    2: "V_CMP_EQ_F32",
    3: "V_CMP_LE_F32",
    4: "V_CMP_GT_F32",
    5: "V_CMP_LG_F32",
    6: "V_CMP_GE_F32",
    7: "V_CMP_O_F32",
    8: "V_CMP_U_F32",
    9: "V_CMP_NGE_F32",
    10: "V_CMP_NLG_F32",
    11: "V_CMP_NGT_F32",
    12: "V_CMP_NLE_F32",
    13: "V_CMP_NEQ_F32",
    14: "V_CMP_NLT_F32",
    15: "V_CMP_TRU_F32",
    16: "V_CMPX_F_F32",
    17: "V_CMPX_LT_F32",
    18: "V_CMPX_EQ_F32",
    19: "V_CMPX_LE_F32",
    20: "V_CMPX_GT_F32",
    21: "V_CMPX_LG_F32",
    22: "V_CMPX_GE_F32",
    23: "V_CMPX_O_F32",
    24: "V_CMPX_U_F32",
    25: "V_CMPX_NGE_F32",
    26: "V_CMPX_NLG_F32",
    27: "V_CMPX_NGT_F32",
    28: "V_CMPX_NLE_F32",
    29: "V_CMPX_NEQ_F32",
    30: "V_CMPX_NLT_F32",
    31: "V_CMPX_TRU_F32",
    32: "V_CMP_F_F64",
    33: "V_CMP_LT_F64",
    34: "V_CMP_EQ_F64",
    35: "V_CMP_LE_F64",
    36: "V_CMP_GT_F64",
    37: "V_CMP_LG_F64",
    38: "V_CMP_GE_F64",
    39: "V_CMP_O_F64",
    40: "V_CMP_U_F64",
    41: "V_CMP_NGE_F64",
    42: "V_CMP_NLG_F64",
    43: "V_CMP_NGT_F64",
    44: "V_CMP_NLE_F64",
    45: "V_CMP_NEQ_F64",
    46: "V_CMP_NLT_F64",
    47: "V_CMP_TRU_F64",
    48: "V_CMPX_F_F64",
    49: "V_CMPX_LT_F64",
    50: "V_CMPX_EQ_F64",
    51: "V_CMPX_LE_F64",
    52: "V_CMPX_GT_F64",
    53: "V_CMPX_LG_F64",
    54: "V_CMPX_GE_F64",
    55: "V_CMPX_O_F64",
    56: "V_CMPX_U_F64",
    57: "V_CMPX_NGE_F64",
    58: "V_CMPX_NLG_F64",
    59: "V_CMPX_NGT_F64",
    60: "V_CMPX_NLE_F64",
    61: "V_CMPX_NEQ_F64",
    62: "V_CMPX_NLT_F64",
    63: "V_CMPX_TRU_F64",
    128: "V_CMP_F_I32",
    129: "V_CMP_LT_I32",
    130: "V_CMP_EQ_I32",
    131: "V_CMP_LE_I32",
    132: "V_CMP_GT_I32",
    133: "V_CMP_NE_I32",
    134: "V_CMP_GE_I32",
    135: "V_CMP_T_I32",
    136: "V_CMP_CLASS_F32",
    137: "V_CMP_LT_I16",
    138: "V_CMP_EQ_I16",
    139: "V_CMP_LE_I16",
    140: "V_CMP_GT_I16",
    141: "V_CMP_NE_I16",
    142: "V_CMP_GE_I16",
    143: "V_CMP_CLASS_F16",
    144: "V_CMPX_F_I32",
    145: "V_CMPX_LT_I32",
    146: "V_CMPX_EQ_I32",
    147: "V_CMPX_LE_I32",
    148: "V_CMPX_GT_I32",
    149: "V_CMPX_NE_I32",
    150: "V_CMPX_GE_I32",
    151: "V_CMPX_T_I32",
    152: "V_CMPX_CLASS_F32",
    153: "V_CMPX_LT_I16",
    154: "V_CMPX_EQ_I16",
    155: "V_CMPX_LE_I16",
    156: "V_CMPX_GT_I16",
    157: "V_CMPX_NE_I16",
    158: "V_CMPX_GE_I16",
    159: "V_CMPX_CLASS_F16",
    160: "V_CMP_F_I64",
    161: "V_CMP_LT_I64",
    162: "V_CMP_EQ_I64",
    163: "V_CMP_LE_I64",
    164: "V_CMP_GT_I64",
    165: "V_CMP_NE_I64",
    166: "V_CMP_GE_I64",
    167: "V_CMP_T_I64",
    168: "V_CMP_CLASS_F64",
    169: "V_CMP_LT_U16",
    170: "V_CMP_EQ_U16",
    171: "V_CMP_LE_U16",
    172: "V_CMP_GT_U16",
    173: "V_CMP_NE_U16",
    174: "V_CMP_GE_U16",
    176: "V_CMPX_F_I64",
    177: "V_CMPX_LT_I64",
    178: "V_CMPX_EQ_I64",
    179: "V_CMPX_LE_I64",
    180: "V_CMPX_GT_I64",
    181: "V_CMPX_NE_I64",
    182: "V_CMPX_GE_I64",
    183: "V_CMPX_T_I64",
    184: "V_CMPX_CLASS_F64",
    185: "V_CMPX_LT_U16",
    186: "V_CMPX_EQ_U16",
    187: "V_CMPX_LE_U16",
    188: "V_CMPX_GT_U16",
    189: "V_CMPX_NE_U16",
    190: "V_CMPX_GE_U16",
    192: "V_CMP_F_U32",
    193: "V_CMP_LT_U32",
    194: "V_CMP_EQ_U32",
    195: "V_CMP_LE_U32",
    196: "V_CMP_GT_U32",
    197: "V_CMP_NE_U32",
    198: "V_CMP_GE_U32",
    199: "V_CMP_T_U32",
    200: "V_CMP_F_F16",
    201: "V_CMP_LT_F16",
    202: "V_CMP_EQ_F16",
    203: "V_CMP_LE_F16",
    204: "V_CMP_GT_F16",
    205: "V_CMP_LG_F16",
    206: "V_CMP_GE_F16",
    207: "V_CMP_O_F16",
    208: "V_CMPX_F_U32",
    209: "V_CMPX_LT_U32",
    210: "V_CMPX_EQ_U32",
    211: "V_CMPX_LE_U32",
    212: "V_CMPX_GT_U32",
    213: "V_CMPX_NE_U32",
    214: "V_CMPX_GE_U32",
    215: "V_CMPX_T_U32",
    216: "V_CMPX_F_F16",
    217: "V_CMPX_LT_F16",
    218: "V_CMPX_EQ_F16",
    219: "V_CMPX_LE_F16",
    220: "V_CMPX_GT_F16",
    221: "V_CMPX_LG_F16",
    222: "V_CMPX_GE_F16",
    223: "V_CMPX_O_F16",
    224: "V_CMP_F_U64",
    225: "V_CMP_LT_U64",
    226: "V_CMP_EQ_U64",
    227: "V_CMP_LE_U64",
    228: "V_CMP_GT_U64",
    229: "V_CMP_NE_U64",
    230: "V_CMP_GE_U64",
    231: "V_CMP_T_U64",
    232: "V_CMP_U_F16",
    233: "V_CMP_NGE_F16",
    234: "V_CMP_NLG_F16",
    235: "V_CMP_NGT_F16",
    236: "V_CMP_NLE_F16",
    237: "V_CMP_NEQ_F16",
    238: "V_CMP_NLT_F16",
    239: "V_CMP_TRU_F16",
    240: "V_CMPX_F_U64",
    241: "V_CMPX_LT_U64",
    242: "V_CMPX_EQ_U64",
    243: "V_CMPX_LE_U64",
    244: "V_CMPX_GT_U64",
    245: "V_CMPX_NE_U64",
    246: "V_CMPX_GE_U64",
    247: "V_CMPX_T_U64",
    248: "V_CMPX_U_F16",
    249: "V_CMPX_NGE_F16",
    250: "V_CMPX_NLG_F16",
    251: "V_CMPX_NGT_F16",
    252: "V_CMPX_NLE_F16",
    253: "V_CMPX_NEQ_F16",
    254: "V_CMPX_NLT_F16",
    255: "V_CMPX_TRU_F16",
}
VOP3 = {
    320: "V_FMA_LEGACY_F32",
    322: "V_MAD_I32_I24",
    323: "V_MAD_U32_U24",
    324: "V_CUBEID_F32",
    325: "V_CUBESC_F32",
    326: "V_CUBETC_F32",
    327: "V_CUBEMA_F32",
    328: "V_BFE_U32",
    329: "V_BFE_I32",
    330: "V_BFI_B32",
    331: "V_FMA_F32",
    332: "V_FMA_F64",
    333: "V_LERP_U8",
    334: "V_ALIGNBIT_B32",
    335: "V_ALIGNBYTE_B32",
    336: "V_MULLIT_F32",
    337: "V_MIN3_F32",
    338: "V_MIN3_I32",
    339: "V_MIN3_U32",
    340: "V_MAX3_F32",
    341: "V_MAX3_I32",
    342: "V_MAX3_U32",
    343: "V_MED3_F32",
    344: "V_MED3_I32",
    345: "V_MED3_U32",
    346: "V_SAD_U8",
    347: "V_SAD_HI_U8",
    348: "V_SAD_U16",
    349: "V_SAD_U32",
    350: "V_CVT_PK_U8_F32",
    351: "V_DIV_FIXUP_F32",
    352: "V_DIV_FIXUP_F64",
    356: "V_ADD_F64",
    357: "V_MUL_F64",
    358: "V_MIN_F64",
    359: "V_MAX_F64",
    360: "V_LDEXP_F64",
    361: "V_MUL_LO_U32",
    362: "V_MUL_HI_U32",
    364: "V_MUL_HI_I32",
    365: "V_DIV_SCALE_F32",
    366: "V_DIV_SCALE_F64",
    367: "V_DIV_FMAS_F32",
    368: "V_DIV_FMAS_F64",
    369: "V_MSAD_U8",
    370: "V_QSAD_PK_U16_U8",
    371: "V_MQSAD_PK_U16_U8",
    372: "V_TRIG_PREOP_F64",
    373: "V_MQSAD_U32_U8",
    374: "V_MAD_U64_U32",
    375: "V_MAD_I64_I32",
    376: "V_XOR3_B32",
    767: "V_LSHLREV_B64",
    768: "V_LSHRREV_B64",
    769: "V_ASHRREV_I64",
    771: "V_ADD_NC_U16",
    772: "V_SUB_NC_U16",
    773: "V_MUL_LO_U16",
    775: "V_LSHRREV_B16",
    776: "V_ASHRREV_I16",
    777: "V_MAX_U16",
    778: "V_MAX_I16",
    779: "V_MIN_U16",
    780: "V_MIN_I16",
    781: "V_ADD_NC_I16",
    782: "V_SUB_NC_I16",
    783: "V_ADD_CO_U32",
    784: "V_SUB_CO_U32",
    785: "V_PACK_B32_F16",
    786: "V_CVT_PKNORM_I16_F16",
    787: "V_CVT_PKNORM_U16_F16",
    788: "V_LSHLREV_B16",
    793: "V_SUBREV_CO_U32",
    832: "V_MAD_U16",
    834: "V_INTERP_P1LL_F16",
    835: "V_INTERP_P1LV_F16",
    836: "V_PERM_B32",
    837: "V_XAD_U32",
    838: "V_LSHL_ADD_U32",
    839: "V_ADD_LSHL_U32",
    843: "V_FMA_F16",
    849: "V_MIN3_F16",
    850: "V_MIN3_I16",
    851: "V_MIN3_U16",
    852: "V_MAX3_F16",
    853: "V_MAX3_I16",
    854: "V_MAX3_U16",
    855: "V_MED3_F16",
    856: "V_MED3_I16",
    857: "V_MED3_U16",
    858: "V_INTERP_P2_F16",
    862: "V_MAD_I16",
    863: "V_DIV_FIXUP_F16",
    864: "V_READLANE_B32",
    865: "V_WRITELANE_B32",
    866: "V_LDEXP_F32",
    867: "V_BFM_B32",
    868: "V_BCNT_U32_B32",
    869: "V_MBCNT_LO_U32_B32",
    870: "V_MBCNT_HI_U32_B32",
    872: "V_CVT_PKNORM_I16_F32",
    873: "V_CVT_PKNORM_U16_F32",
    874: "V_CVT_PK_U16_U32",
    875: "V_CVT_PK_I16_I32",
    877: "V_ADD3_U32",
    879: "V_LSHL_OR_B32",
    881: "V_AND_OR_B32",
    882: "V_OR3_B32",
    883: "V_MAD_U32_U16",
    885: "V_MAD_I32_I16",
    886: "V_SUB_NC_I32",
    887: "V_PERMLANE16_B32",
    888: "V_PERMLANEX16_B32",
    895: "V_ADD_NC_I32",
}


# ---------------------------------------------------------------------------
# Format identification (ISA doc ch. 13, "Microcode Formats"; the ENCODING field
# of each format table gives the fixed high bits used below).
# ---------------------------------------------------------------------------

VOP3_ALIASES_VOP3B = {296, 297, 298, 365, 366, 374, 375, 783, 784, 793}


def _bits(word: int, hi: int, lo: int) -> int:
    """Extract bits [hi:lo] of ``word``."""
    return (word >> lo) & ((1 << (hi - lo + 1)) - 1)


@dataclass
class Insn:
    """One decoded instruction: byte offset, byte length, format, mnemonic, raw fields."""

    pc: int
    size: int
    fmt: str
    op: int
    name: str | None  # upper-case ISA name; None when this tool has no opcode table for it
    fields: dict = field(default_factory=dict)
    words: tuple = ()

    @property
    def mnemonic(self) -> str:
        """Lower-case mnemonic, or ``<fmt>?`` when the opcode table does not cover it."""
        return self.name.lower() if self.name else f"<{self.fmt.lower()}?>"


class DecodeError(Exception):
    """Raised when the byte stream cannot be decoded to an instruction boundary."""


def _src_has_literal(code: int) -> bool:
    """True if a 9-bit/8-bit source code is followed by a 32-bit extra dword.

    255 = literal constant; 233/234 = DPP8/DPP8FI, 249 = SDWA, 250 = DPP16 (the
    VOP1/2/C source-0 extension dword). Section 13.3 source-operand table.
    """
    return code in (255, 233, 234, 249, 250)


def decode_one(data: bytes, off: int) -> Insn:
    """Decode the instruction at byte offset ``off`` of ``data``."""
    if off + 4 > len(data):
        raise DecodeError(f"truncated at {off:#x}")
    w0 = struct.unpack_from("<I", data, off)[0]
    w1 = struct.unpack_from("<I", data, off + 4)[0] if off + 8 <= len(data) else None

    def need2() -> int:
        if w1 is None:
            raise DecodeError(f"truncated 64-bit instruction at {off:#x}")
        return w1

    def mk(fmt, op, name, size, fields):
        if off + size > len(data):
            raise DecodeError(f"truncated {fmt} at {off:#x}")
        words = struct.unpack_from(f"<{size // 4}I", data, off)
        return Insn(off, size, fmt, op, name, fields, words)

    # --- scalar formats: ENCODING[31:30] == 10 -----------------------------
    if _bits(w0, 31, 30) == 0b10:
        top9 = _bits(w0, 31, 23)
        if top9 == 0b101111101:  # SOP1
            op = _bits(w0, 15, 8)
            f = {"ssrc0": _bits(w0, 7, 0), "sdst": _bits(w0, 22, 16)}
            return mk("SOP1", op, SOP1.get(op), 8 if f["ssrc0"] == 255 else 4, f)
        if top9 == 0b101111110:  # SOPC
            op = _bits(w0, 22, 16)
            f = {"ssrc0": _bits(w0, 7, 0), "ssrc1": _bits(w0, 15, 8)}
            lit = 255 in (f["ssrc0"], f["ssrc1"])
            return mk("SOPC", op, SOPC.get(op), 8 if lit else 4, f)
        if top9 == 0b101111111:  # SOPP
            op = _bits(w0, 22, 16)
            return mk("SOPP", op, SOPP.get(op), 4, {"simm16": _bits(w0, 15, 0)})
        if _bits(w0, 31, 28) == 0b1011:  # SOPK
            op = _bits(w0, 27, 23)
            f = {"simm16": _bits(w0, 15, 0), "sdst": _bits(w0, 22, 16)}
            return mk("SOPK", op, SOPK.get(op), 8 if op == 21 else 4, f)
        op = _bits(w0, 29, 23)  # SOP2
        f = {
            "ssrc0": _bits(w0, 7, 0),
            "ssrc1": _bits(w0, 15, 8),
            "sdst": _bits(w0, 22, 16),
        }
        lit = 255 in (f["ssrc0"], f["ssrc1"])
        return mk("SOP2", op, SOP2.get(op), 8 if lit else 4, f)

    # --- VOP2 / VOP1 / VOPC: bit 31 == 0 -----------------------------------
    if _bits(w0, 31, 31) == 0:
        top7 = _bits(w0, 31, 25)
        src0 = _bits(w0, 8, 0)
        ext = 4 if _src_has_literal(src0) else 0
        if top7 == 0b0111111:  # VOP1
            op = _bits(w0, 16, 9)
            f = {"src0": src0, "vdst": _bits(w0, 24, 17)}
            return mk("VOP1", op, VOP1.get(op), 4 + ext, f)
        if top7 == 0b0111110:  # VOPC
            op = _bits(w0, 24, 17)
            f = {"src0": src0, "vsrc1": _bits(w0, 16, 9)}
            return mk("VOPC", op, VOPC.get(op), 4 + ext, f)
        op = _bits(w0, 30, 25)  # VOP2
        f = {"src0": src0, "vsrc1": _bits(w0, 16, 9), "vdst": _bits(w0, 24, 17)}
        # FMAMK/FMAAK (f32 44/45, f16 55/56) always carry a literal dword.
        lit = 4 if op in (44, 45, 55, 56) else ext
        return mk("VOP2", op, VOP2.get(op), 4 + lit, f)

    # --- 64-bit / other formats: ENCODING[31:26] ---------------------------
    enc6 = _bits(w0, 31, 26)
    if enc6 == 0b110101:  # VOP3 (VOP3A / VOP3B share the encoding)
        w1 = need2()
        op = _bits(w0, 25, 16)
        is_vop3b = op in VOP3_ALIASES_VOP3B
        fmt = "VOP3B" if is_vop3b else "VOP3"
        f = {
            "vdst": _bits(w0, 7, 0),
            "sdst": _bits(w0, 14, 8) if is_vop3b else 0,
            "opsel": 0 if is_vop3b else _bits(w0, 14, 11),
            "abs": 0 if is_vop3b else _bits(w0, 10, 8),
            "clamp": _bits(w0, 15, 15),
            "src0": _bits(w1, 8, 0),
            "src1": _bits(w1, 17, 9),
            "src2": _bits(w1, 26, 18),
            "omod": 0 if is_vop3b else _bits(w1, 28, 27),
            "neg": 0 if is_vop3b else _bits(w1, 31, 29),
        }
        if op < 256:
            name, sub = VOPC.get(op), "VOPC"
        elif op < 320:
            name, sub = VOP2.get(op - 256), "VOP2"
        elif 384 <= op < 512:
            name, sub = VOP1.get(op - 384), "VOP1"
        else:
            name, sub = VOP3.get(op), "VOP3"
        f["base"] = sub
        lit = any(f[k] == 255 for k in ("src0", "src1", "src2"))
        return mk(fmt, op, name, 12 if lit else 8, f)
    if enc6 == 0b111101:  # SMEM
        w1 = need2()
        op = _bits(w0, 25, 18)
        f = {
            "sbase": _bits(w0, 5, 0) * 2,
            "sdata": _bits(w0, 12, 6),
            "glc": _bits(w0, 16, 16),
            "dlc": _bits(w0, 14, 14),
            "offset": _bits(w1, 20, 0),
            "soffset": _bits(w1, 31, 25),
        }
        return mk("SMEM", op, SMEM.get(op), 8, f)
    if enc6 == 0b111110:  # EXP
        return mk("EXP", 0, "EXP", 8, {})
    other = {
        0b110011: "VOP3P",
        0b110010: "VINTRP",
        0b110110: "DS",
        0b110111: "FLAT",
        0b111000: "MUBUF",
        0b111010: "MTBUF",
        0b111100: "MIMG",
    }
    if enc6 in other:
        fmt = other[enc6]
        if fmt == "MIMG" and _bits(w0, 0, 0):
            # Non-sequential-address MIMG appends address dwords whose count depends on
            # the dimension; this decoder refuses to guess the length.
            raise DecodeError(f"MIMG NSA at {off:#x}: length not modelled")
        if fmt == "VOP3P":
            w1 = need2()
            lit = any(_bits(w1, hi, lo) == 255 for hi, lo in ((8, 0), (17, 9), (26, 18)))
            return mk(fmt, _bits(w0, 22, 16), None, 12 if lit else 8, {})
        return mk(fmt, 0, None, 8, {})
    raise DecodeError(f"unknown encoding {w0:#010x} at {off:#x}")


def decode_program(data: bytes) -> tuple[list[Insn], str | None]:
    """Decode linearly from offset 0. Returns (instructions, error-or-None)."""
    out: list[Insn] = []
    off = 0
    while off < len(data):
        try:
            insn = decode_one(data, off)
        except DecodeError as exc:
            return out, str(exc)
        out.append(insn)
        off += insn.size
    return out, None
