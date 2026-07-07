// agc_reg_defaults.cpp — the RegisterDefaults tables that libSceAgc's
// sceAgcGetRegisterDefaults2 (NID 2JtWUUiYBXs) / GetRegisterDefaults (wRbq6ZjNop4) return.
//
// The tables (register hashes, PM4 register offsets, default values) and the PM4 register
// constants are VENDORED VERBATIM from Kyty (MIT License, Copyright (c) 2021 Ivan Chikhradze),
// files source/emulator/src/Graphics/Graphics.cpp and include/Emulator/Graphics/Pm4.h. Kyty
// reverse-engineered the exact layout the PS5 AGC SDK expects. prosper reuses it so the game's
// Gen5 graphics init (Unity GfxDevicePS5) can build its internal register-offset table from a
// real, non-empty RegisterDefaults instead of the previous count=0 placeholder.
//
// Consumed by hle_graphics.cpp's GetRegisterDefaults2 thunk via prosper_agc_reg_defaults().
#include <cstddef>
#include <cstdint>

#include "../gpu/pm4_registers.hpp"

namespace prosper { namespace agc {

struct ShaderRegister { uint32_t offset; uint32_t value; };
struct RegisterDefaultInfo { uint32_t type; ShaderRegister reg[16]; };
struct RegisterDefaults {
    ShaderRegister** tbl0       = nullptr;
    ShaderRegister** tbl1       = nullptr;
    ShaderRegister** tbl2       = nullptr;
    ShaderRegister** tbl3       = nullptr;
    uint64_t         unknown[2] = {};
    uint32_t*        types      = nullptr;
    uint32_t         count      = 0;
};
static_assert(offsetof(RegisterDefaults, count) == 0x38, "AGC RegisterDefaults layout must match SDK");

static RegisterDefaultInfo g_cx_reg_info1[] = {
    /* 0 */ {0xE24F806D, {{Pm4::CB_COLOR_CONTROL, 0x00cc0010}}},
    /* 1 */ {0xF6C28182, {{Pm4::CB_DCC_CONTROL, 0x00000000}}},
    /* 2 */ {0x6F6E55A5, {{Pm4::CB_RMI_GL2_CACHE_CONTROL, 0x00000000}}},
    /* 3 */ {0x0BC65DA4, {{Pm4::CB_SHADER_MASK, 0x00000000}}},
    /* 4 */ {0x9E5AD592, {{Pm4::CB_TARGET_MASK, 0x0000000F}}},
    /* 5 */ {0xBB513B98, {{Pm4::DB_ALPHA_TO_MASK, 0x0000aa00}}},
    /* 6 */ {0xAB64B23B, {{Pm4::DB_COUNT_CONTROL, 0x00000000}}},
    /* 7 */ {0x53C39964, {{Pm4::DB_DEPTH_CONTROL, 0x00000000}}},
    /* 8 */ {0x01396B11, {{Pm4::DB_EQAA, 0x00000000}}},
    /* 9 */ {0x7D42019A, {{Pm4::DB_RENDER_CONTROL, 0x00000000}}},
    /* 10 */ {0x3548F523, {{Pm4::PS_SHADER_SAMPLE_EXCLUSION_MASK, 0x00000000}}},
    /* 11 */ {0xF43AD28A, {{Pm4::DB_RMI_L2_CACHE_CONTROL, 0x00000000}}},
    /* 12 */ {0x6DE4C312, {{Pm4::DB_SHADER_CONTROL, 0x00000000}}},
    /* 13 */ {0x00A77AE0, {{Pm4::DB_SRESULTS_COMPARE_STATE0, 0x00000000}}},
    /* 14 */ {0x00A779B7, {{Pm4::DB_SRESULTS_COMPARE_STATE1, 0x00000000}}},
    /* 15 */ {0x5100100C, {{Pm4::DB_STENCILREFMASK, 0x00000000}}},
    /* 16 */ {0x59958BBA, {{Pm4::DB_STENCILREFMASK_BF, 0x00000000}}},
    /* 17 */ {0x0C06F17C, {{Pm4::DB_STENCIL_CONTROL, 0x00000000}}},
    /* 18 */ {0x6F104B72, {{Pm4::GE_MAX_OUTPUT_PER_SUBGROUP, 0x00000000}}},
    /* 19 */ {0x25C70D9C, {{Pm4::PA_CL_CLIP_CNTL, 0x00000000}}},
    /* 20 */ {0x3881201E, {{Pm4::PA_CL_OBJPRIM_ID_CNTL, 0x00000000}}},
    /* 21 */ {0x09AFDDAF, {{Pm4::PA_CL_VTE_CNTL, 0x0000043f}}},
    /* 22 */ {0x367D63CF, {{Pm4::PA_SC_AA_CONFIG, 0x00000000}}},
    /* 23 */ {0x43707DB8, {{Pm4::PA_SC_CLIPRECT_RULE, 0x0000ffff}}},
    /* 24 */ {0xF6AE26BA, {{Pm4::PA_SC_CONSERVATIVE_RASTERIZATION_CNTL, 0x00000000}}},
    /* 25 */ {0x1B917652, {{Pm4::PA_SC_FSR_ENABLE, 0x00000000}}},
    /* 26 */ {0x94B1E4F7, {{Pm4::PA_SC_HORIZ_GRID, 0x00000000}}},
    /* 27 */ {0xE3661B6C, {{Pm4::PA_SC_LEFT_VERT_GRID, 0x00000000}}},
    /* 28 */ {0x1EB8D73A, {{Pm4::PA_SC_MODE_CNTL_0, 0x00000002}}},
    /* 29 */ {0x15051FA3, {{Pm4::PA_SC_MODE_CNTL_1, 0x00000000}}},
    /* 30 */ {0x9C51A7F1, {{Pm4::PA_SC_RIGHT_VERT_GRID, 0x00000000}}},
    /* 31 */ {0xA20EFC70, {{Pm4::PA_SC_WINDOW_OFFSET, 0x00000000}}},
    /* 32 */ {0x0EC09F6E, {{Pm4::PA_STATE_STEREO_X, 0x00000000}}},
    /* 33 */ {0x34A7D6D3, {{Pm4::PA_STEREO_CNTL, 0x00000000}}},
    /* 34 */ {0xCE831B94, {{Pm4::PA_SU_HARDWARE_SCREEN_OFFSET, 0x00000000}}},
    /* 35 */ {0x5CC72A74, {{Pm4::PA_SU_LINE_CNTL, 0x00000008}}},
    /* 36 */ {0x3B77713C, {{Pm4::PA_SU_POINT_MINMAX, 0xffff0000}}},
    /* 37 */ {0x40F64410, {{Pm4::PA_SU_POINT_SIZE, 0x00080008}}},
    /* 38 */ {0x69441268, {{Pm4::PA_SU_POLY_OFFSET_CLAMP, 0x00000000}}},
    /* 39 */ {0x2E418B83, {{Pm4::PA_SU_POLY_OFFSET_DB_FMT_CNTL, 0x000001e9}}},
    /* 40 */ {0xA00D0C8D, {{Pm4::PA_SU_SC_MODE_CNTL, 0x00000240}}},
    /* 41 */ {0xB1289FB3, {{Pm4::PA_SU_SMALL_PRIM_FILTER_CNTL, 0x00000001}}},
    /* 42 */ {0x144832FB, {{Pm4::PA_SU_VTX_CNTL, 0x0000002d}}},
    /* 43 */ {0x9890D9FA, {{Pm4::SPI_TMPRING_SIZE, 0x00000000}}},
    /* 44 */ {0x9016FAF1, {{Pm4::VGT_DRAW_PAYLOAD_CNTL, 0x00000000}}},
    /* 45 */ {0x4B73CE27, {{Pm4::VGT_GS_MAX_VERT_OUT, 0x00000400}}},
    /* 46 */ {0x5F5A3E7B, {{Pm4::VGT_GS_OUT_PRIM_TYPE, 0x00000002}}},
    /* 47 */ {0xD4AF3A51, {{Pm4::VGT_LS_HS_CONFIG, 0x00000000}}},
    /* 48 */ {0x6CF4F543, {{Pm4::VGT_PRIMITIVEID_RESET, 0xffffffff}}},
    /* 49 */ {0x5FB86CCB, {{Pm4::VGT_PRIMITIVEID_EN, 0x00000000}}},
    /* 50 */ {0xEDEFA188, {{Pm4::VGT_REUSE_OFF, 0x00000000}}},
    /* 51 */ {0xD0DE9EE6, {{Pm4::VGT_SHADER_STAGES_EN, 0x00000000}}},
    /* 52 */ {0xC5831803, {{Pm4::VGT_TESS_DISTRIBUTION, 0x88101000}}},
    /* 53 */ {0x8E6DE84B, {{Pm4::VGT_TF_PARAM, 0x00000000}}},
    /* 54 */
    {0xD0771662,
     {
         {Pm4::PA_SC_CENTROID_PRIORITY_0, 0x00000000},
         {Pm4::PA_SC_CENTROID_PRIORITY_1, 0x00000000},
     }},
    /* 55 */ {0x569F7444, {{Pm4::PA_SC_AA_SAMPLE_LOCS_PIXEL_X0Y0_0, 0x00000000}}},
    /* 56 */
    {0x5C6637CD,
     {
         {Pm4::PA_SC_AA_MASK_X0Y0_X1Y0, 0xffffffff},
         {Pm4::PA_SC_AA_MASK_X0Y1_X1Y1, 0xffffffff},
     }},
    /* 57 */
    {0xCAE3E690,
     {
         {Pm4::PA_SC_BINNER_CNTL_0, 0x00000002},
         {Pm4::PA_SC_BINNER_CNTL_1, 0x03ff0080},
     }},
    /* 58 */
    {0x43FBD769,
     {
         {Pm4::CB_BLEND_RED, 0x00000000},
         {Pm4::CB_BLEND_BLUE, 0x00000000},
         {Pm4::CB_BLEND_GREEN, 0x00000000},
         {Pm4::CB_BLEND_ALPHA, 0x00000000},
     }},
    /* 59 */ {0xEF550356, {{Pm4::CB_BLEND0_CONTROL, 0x20010001}}},
    /* 60 */
    {0x8F52E279,
     {
         {Pm4::TA_BC_BASE_ADDR, 0x00000000},
         {Pm4::TA_BC_BASE_ADDR_HI, 0x00000000},
     }},
    /* 61 */
    {0x1F2D8149,
     {
         {Pm4::PA_SC_CLIPRECT_0_TL, 0x00000000},
         {Pm4::PA_SC_CLIPRECT_0_BR, 0x20002000},
     }},
    /* 62 */ {0x853D0614, {{Pm4::CX_NOP, 0x00000000}}},
    /* 63 */
    {0x4413C6F9,
     {
         {Pm4::DB_DEPTH_BOUNDS_MIN, 0x00000000},
         {Pm4::DB_DEPTH_BOUNDS_MAX, 0x00000000},
     }},
    /* 64 */
    {0x67096014,
     {
         {Pm4::DB_Z_INFO, 0x80000000},
         {Pm4::DB_STENCIL_INFO, 0x20000000},
         {Pm4::DB_Z_READ_BASE, 0x00000000},
         {Pm4::DB_STENCIL_READ_BASE, 0x00000000},
         {Pm4::DB_Z_WRITE_BASE, 0x00000000},
         {Pm4::DB_STENCIL_WRITE_BASE, 0x00000000},
         {Pm4::DB_Z_READ_BASE_HI, 0x00000000},
         {Pm4::DB_STENCIL_READ_BASE_HI, 0x00000000},
         {Pm4::DB_Z_WRITE_BASE_HI, 0x00000000},
         {Pm4::DB_STENCIL_WRITE_BASE_HI, 0x00000000},
         {Pm4::DB_HTILE_DATA_BASE_HI, 0x00000000},
         {Pm4::DB_DEPTH_VIEW, 0x00000000},
         {Pm4::DB_HTILE_DATA_BASE, 0x00000000},
         {Pm4::DB_DEPTH_SIZE_XY, 0x00000000},
         {Pm4::DB_DEPTH_CLEAR, 0x00000000},
         {Pm4::DB_STENCIL_CLEAR, 0x00000000},
     }},
    /* 65 */
    {0x88F5E915,
     {
         {Pm4::PA_SC_FOV_WINDOW_LR, 0xff00ff00},
         {Pm4::PA_SC_FOV_WINDOW_TB, 0x00000000},
     }},
    /* 66 */
    {0x033F1EFF,
     {
         {Pm4::FSR_RECURSIONS0, 0x00000000},
         {Pm4::FSR_RECURSIONS1, 0x00000000},
     }},
    /* 67 */
    {0x918106BB,
     {
         {Pm4::PA_SC_GENERIC_SCISSOR_TL, 0x80000000},
         {Pm4::PA_SC_GENERIC_SCISSOR_BR, 0x40004000},
     }},
    /* 68 */
    {0x95F0E7AC,
     {
         {Pm4::PA_CL_GB_VERT_CLIP_ADJ, 0x4e7e0000},
         {Pm4::PA_CL_GB_VERT_DISC_ADJ, 0x4e7e0000},
         {Pm4::PA_CL_GB_HORZ_CLIP_ADJ, 0x4e7e0000},
         {Pm4::PA_CL_GB_HORZ_DISC_ADJ, 0x4e7e0000},
     }},
    /* 69 */
    {0xB48CBAB2,
     {
         {Pm4::PA_SU_POLY_OFFSET_BACK_SCALE, 0x00000000},
         {Pm4::PA_SU_POLY_OFFSET_BACK_OFFSET, 0x00000000},
     }},
    /* 70 */
    {0x05BB3BC6,
     {
         {Pm4::PA_SU_POLY_OFFSET_FRONT_SCALE, 0x00000000},
         {Pm4::PA_SU_POLY_OFFSET_FRONT_OFFSET, 0x00000000},
     }},
    /* 71 */
    {0x94FABA07,
     {
         {Pm4::DB_RENDER_OVERRIDE, 0x00000000},
         {Pm4::DB_RENDER_OVERRIDE2, 0x00000000},
     }},
    /* 72 */
    {0x38E92C91,
     {
         {Pm4::CB_COLOR0_BASE, 0x00000000},
         {Pm4::CB_COLOR0_VIEW, 0x00000000},
         {Pm4::CB_COLOR0_INFO, 0x00000000},
         {Pm4::CB_COLOR0_ATTRIB, 0x00000000},
         {Pm4::CB_COLOR0_DCC_CONTROL, 0x00000048},
         {Pm4::CB_COLOR0_CMASK, 0x00000000},
         {Pm4::CB_COLOR0_FMASK, 0x00000000},
         {Pm4::CB_COLOR0_CLEAR_WORD0, 0x00000000},
         {Pm4::CB_COLOR0_CLEAR_WORD1, 0x00000000},
         {Pm4::CB_COLOR0_DCC_BASE, 0x00000000},
         {Pm4::CB_COLOR0_BASE_EXT, 0x00000000},
         {Pm4::CB_COLOR0_CMASK_BASE_EXT, 0x00000000},
         {Pm4::CB_COLOR0_FMASK_BASE_EXT, 0x00000000},
         {Pm4::CB_COLOR0_DCC_BASE_EXT, 0x00000000},
         {Pm4::CB_COLOR0_ATTRIB2, 0x00000000},
         {Pm4::CB_COLOR0_ATTRIB3, 0x0006c000},
     }},
    /* 73 */
    {0x0B177B43,
     {
         {Pm4::PA_SC_SCREEN_SCISSOR_TL, 0x00000000},
         {Pm4::PA_SC_SCREEN_SCISSOR_BR, 0x40004000},
     }},
    /* 74 */ {0x48531062, {{Pm4::SPI_PS_INPUT_CNTL_0, 0x00000000}}},
    /* 75 */
    {0xAAA964B9,
     {
         {Pm4::PA_CL_UCP_0_X, 0x00000000},
         {Pm4::PA_CL_UCP_0_Y, 0x00000000},
         {Pm4::PA_CL_UCP_0_Z, 0x00000000},
         {Pm4::PA_CL_UCP_0_W, 0x00000000},
     }},
    /* 76 */
    {0x7690AF6F,
     {
         {Pm4::PA_CL_VPORT_XSCALE, 0x4e7e0000},
         {Pm4::PA_CL_VPORT_YSCALE, 0x4e7e0000},
         {Pm4::PA_CL_VPORT_ZSCALE, 0x4e7e0000},
         {Pm4::PA_CL_VPORT_XOFFSET, 0x00000000},
         {Pm4::PA_CL_VPORT_YOFFSET, 0x00000000},
         {Pm4::PA_CL_VPORT_ZOFFSET, 0x00000000},
         {Pm4::PA_SC_VPORT_SCISSOR_0_TL, 0x80000000},
         {Pm4::PA_SC_VPORT_SCISSOR_0_BR, 0x40004000},
         {Pm4::PA_SC_VPORT_ZMIN_0, 0x00000000},
         {Pm4::PA_SC_VPORT_ZMAX_0, 0x00000000},
     }},
    /* 77 */
    {0x078D7060,
     {
         {Pm4::PA_SC_WINDOW_SCISSOR_TL, 0x80000000},
         {Pm4::PA_SC_WINDOW_SCISSOR_BR, 0x40004000},
     }},

};

static RegisterDefaultInfo g_sh_reg_info1[] = {
    /* 0 */ {0x5D6E3EC7, {{Pm4::COMPUTE_PGM_RSRC1, 0x00000000}}},
    /* 1 */ {0x57E7079A, {{Pm4::COMPUTE_PGM_RSRC2, 0x00000000}}},
    /* 2 */ {0x7467FAFD, {{Pm4::COMPUTE_PGM_RSRC3, 0x00000000}}},
    /* 3 */ {0x9E826B50, {{Pm4::COMPUTE_RESOURCE_LIMITS, 0x00000000}}},
    /* 4 */ {0xDC484F18, {{Pm4::COMPUTE_TMPRING_SIZE, 0x00000000}}},
    /* 5 */ {0x5DA8BCA3, {{Pm4::SPI_SHADER_PGM_RSRC1_GS, 0x00000000}}},
    /* 6 */ {0x5CA726D8, {{Pm4::SPI_SHADER_PGM_RSRC1_HS, 0x00000000}}},
    /* 7 */ {0x5DD28360, {{Pm4::SPI_SHADER_PGM_RSRC1_PS, 0x00000000}}},
    /* 8 */ {0x57EFA0BE, {{Pm4::SPI_SHADER_PGM_RSRC2_GS, 0x00000000}}},
    /* 9 */ {0x502363D5, {{Pm4::SPI_SHADER_PGM_RSRC2_HS, 0x00000000}}},
    /* 10 */ {0x506D14BD, {{Pm4::SPI_SHADER_PGM_RSRC2_PS, 0x00000000}}},
    /* 11 */ {0xB2609506, {{Pm4::COMPUTE_USER_ACCUM_0, 0x00000000}}},
    /* 12 */
    {0x9E5CFB8A,
     {
         {Pm4::SPI_SHADER_PGM_RSRC3_HS, 0x00000000},
         {Pm4::SPI_SHADER_PGM_RSRC3_GS, 0x00000000},
         {Pm4::SPI_SHADER_PGM_RSRC3_PS, 0x00000000},
     }},
    /* 13 */
    {0xC918DF3E,
     {
         {Pm4::COMPUTE_PGM_LO, 0x00000000},
         {Pm4::COMPUTE_PGM_HI, 0x00000000},
     }},
    /* 14 */
    {0xC9751C9C,
     {
         {Pm4::SPI_SHADER_PGM_LO_ES, 0x00000000},
         {Pm4::SPI_SHADER_PGM_HI_ES, 0x00000000},
     }},
    /* 15 */
    {0xC97EF77A,
     {
         {Pm4::SPI_SHADER_PGM_LO_GS, 0x00000000},
         {Pm4::SPI_SHADER_PGM_HI_GS, 0x00000000},
     }},
    /* 16 */
    {0xC927C6B9,
     {
         {Pm4::SPI_SHADER_PGM_LO_HS, 0x00000000},
         {Pm4::SPI_SHADER_PGM_HI_HS, 0x00000000},
     }},
    /* 17 */
    {0xC92A1EC5,
     {
         {Pm4::SPI_SHADER_PGM_LO_LS, 0x00000000},
         {Pm4::SPI_SHADER_PGM_HI_LS, 0x00000000},
     }},
    /* 18 */
    {0xC9E01B31,
     {
         {Pm4::SPI_SHADER_PGM_LO_PS, 0x00000000},
         {Pm4::SPI_SHADER_PGM_HI_PS, 0x00000000},
     }},
    /* 19 */ {0x50685F29, {{Pm4::SH_NOP, 0x00000000}}},
    /* 20 */ {0xB26219CA, {{Pm4::SPI_SHADER_USER_ACCUM_ESGS_0, 0x00000000}}},
    /* 21 */ {0xB25B6CF9, {{Pm4::SPI_SHADER_USER_ACCUM_LSHS_0, 0x00000000}}},
    /* 22 */ {0xB2F86101, {{Pm4::SPI_SHADER_USER_ACCUM_PS_0, 0x00000000}}},
    /* 23 */
    {0x07E3B155,
     {
         {Pm4::SPI_SHADER_USER_DATA_ADDR_LO_GS, 0x00000000},
         {Pm4::SPI_SHADER_USER_DATA_ADDR_HI_GS, 0x00000000},
     }},
    /* 24 */
    {0x07E383C6,
     {
         {Pm4::SPI_SHADER_USER_DATA_ADDR_LO_HS, 0x00000000},
         {Pm4::SPI_SHADER_USER_DATA_ADDR_HI_HS, 0x00000000},
     }},
    /* 25 */ {0xBDA98653, {{Pm4::COMPUTE_USER_DATA_0, 0x00000000}}},
    /* 26 */ {0xBDBD1D0F, {{Pm4::SPI_SHADER_USER_DATA_GS_0, 0x00000000}}},
    /* 27 */ {0xBD946FD4, {{Pm4::SPI_SHADER_USER_DATA_HS_0, 0x00000000}}},
    /* 28 */ {0xBDF02A4C, {{Pm4::SPI_SHADER_USER_DATA_PS_0, 0x00000000}}},
};

static RegisterDefaultInfo g_uc_reg_info1[] = {
    /* 0 */ {0x19E93E85, {{Pm4::GDS_OA_ADDRESS, 0x00000000}}},
    /* 1 */ {0x3B5C2AF3, {{Pm4::GDS_OA_CNTL, 0x00000000}}},
    /* 2 */ {0x47974A35, {{Pm4::GDS_OA_COUNTER, 0x00000000}}},
    /* 3 */ {0x105971C2, {{Pm4::GE_CNTL, 0x00000000}}},
    /* 4 */ {0x7D137765, {{Pm4::GE_INDX_OFFSET, 0x00000000}}},
    /* 5 */ {0xD187FEBC, {{Pm4::GE_MULTI_PRIM_IB_RESET_EN, 0x00000000}}},
    /* 6 */ {0x12F854AC, {{Pm4::GE_STEREO_CNTL, 0x00000000}}},
    /* 7 */ {0x40D49AD1, {{Pm4::GE_USER_VGPR_EN, 0x00000000}}},
    /* 8 */ {0x8C0923DA, {{Pm4::FSR_EXTEND_SUBPIXEL_ROUNDING, 0x00000000}}},
    /* 9 */ {0xBB8DF494, {{Pm4::TEXTURE_GRADIENT_CONTROL, 0x00000000}}},
    /* 10 */ {0xF6D8A76E, {{Pm4::TEXTURE_GRADIENT_FACTORS, 0x40000040}}},
    /* 11 */ {0x7620F1E9, {{Pm4::VGT_OBJECT_ID, 0x00000000}}},
    /* 12 */ {0x9EBFAB10, {{Pm4::VGT_PRIMITIVE_TYPE, 0x00000000}}},
    /* 13 */
    {0x98A09D0E,
     {
         {Pm4::TA_CS_BC_BASE_ADDR, 0x00000000},
         {Pm4::TA_CS_BC_BASE_ADDR_HI, 0x00000000},
     }},
    /* 14 */
    {0x195D37D2,
     {
         {Pm4::FSR_ALPHA_VALUE0, 0x00000000},
         {Pm4::FSR_ALPHA_VALUE1, 0x00000000},
     }},
    /* 15 */
    {0xF9EC4F85,
     {
         {Pm4::FSR_CONTROL_POINT0, 0x00000000},
         {Pm4::FSR_CONTROL_POINT1, 0x00000000},
         {Pm4::FSR_CONTROL_POINT2, 0x00000000},
         {Pm4::FSR_CONTROL_POINT3, 0x00000000},
     }},
    /* 16 */
    {0x4626B750,
     {
         {Pm4::FSR_WINDOW0, 0x00000000},
         {Pm4::FSR_WINDOW1, 0x00000000},
     }},
    /* 17 */ {0x4CC673A0, {{Pm4::MEMORY_MAPPING_MASK, 0x00000000}}},
    /* 18 */ {0xDE5B3431, {{Pm4::UC_NOP, 0x00000000}}},
    /* 19 */ {0x036AC8A6, {{Pm4::GE_USER_VGPR1, 0x00000000}}}};

static RegisterDefaultInfo g_cx_reg_info2[] = {
    /* 0 */ {0x8FB4EDB5, {{Pm4::DB_DFSM_CONTROL, 0x00000000}}},
    /* 1 */ {0xB994AD29, {{Pm4::DB_HTILE_SURFACE, 0x00000000}}},
    /* 2 */ {0xD427322F, {{Pm4::PA_SC_NGG_MODE_CNTL, 0x00000000}}},
    /* 3 */ {0xF58FEA31, {{Pm4::SPI_INTERP_CONTROL_0, 0x00000000}}},
};

static RegisterDefaultInfo g_sh_reg_info2[] = {
    /* 0 */ {0x6AC156EF, {{Pm4::COMPUTE_DESTINATION_EN_SE0, 0x00000000}}},
    /* 1 */ {0x6AC15610, {{Pm4::COMPUTE_DESTINATION_EN_SE1, 0x00000000}}},
    /* 2 */ {0x6AC15009, {{Pm4::COMPUTE_DESTINATION_EN_SE2, 0x00000000}}},
    /* 3 */ {0x6AC153BA, {{Pm4::COMPUTE_DESTINATION_EN_SE3, 0x00000000}}},
    /* 4 */ {0xBE7DCD73, {{Pm4::COMPUTE_DISPATCH_TUNNEL, 0x00000000}}},
    /* 5 */ {0x0C4B1438, {{Pm4::COMPUTE_SHADER_CHKSUM, 0x00000000}}},
    /* 6 */ {0xDB00D71A, {{Pm4::COMPUTE_START_X, 0x00000000}}},
    /* 7 */ {0xDB00D249, {{Pm4::COMPUTE_START_Y, 0x00000000}}},
    /* 8 */ {0xDB00EC60, {{Pm4::COMPUTE_START_Z, 0x00000000}}},
    /* 9 */ {0x0C4D6FE4, {{Pm4::SPI_SHADER_PGM_CHKSUM_GS, 0x00000000}}},
    /* 10 */ {0x0C4A80EF, {{Pm4::SPI_SHADER_PGM_CHKSUM_HS, 0x00000000}}},
    /* 11 */ {0x0DD283E7, {{Pm4::SPI_SHADER_PGM_CHKSUM_PS, 0x00000000}}},
    /* 12 */ {0xC620E68C, {{Pm4::SPI_SHADER_PGM_RSRC4_GS, 0x00000000}}},
    /* 13 */ {0xC67EFACF, {{Pm4::SPI_SHADER_PGM_RSRC4_HS, 0x00000000}}},
    /* 14 */ {0xD9E6D9F7, {{Pm4::SPI_SHADER_PGM_RSRC4_PS, 0x00000000}}},
};

static RegisterDefaultInfo g_uc_reg_info2[] = {
    /* 0 */ {0x31F34B9F, {{Pm4::VGT_HS_OFFCHIP_PARAM, 0x00000000}}},
    /* 1 */ {0xAC0F9E76, {{Pm4::UC_NOP, 0x00000000}}},
    /* 2 */ {0x929FD95D, {{Pm4::VGT_TF_MEMORY_BASE, 0x00000000}}},
};

#define KYTY_ID(id, tbl)   ((id)*4 + (tbl))
#define KYTY_INDEX_CX1(id) g_cx_reg_info1[id].type, KYTY_ID(id, 0), 0
#define KYTY_INDEX_SH1(id) g_sh_reg_info1[id].type, KYTY_ID(id, 1), 0
#define KYTY_INDEX_UC1(id) g_uc_reg_info1[id].type, KYTY_ID(id, 2), 0
#define KYTY_INDEX_CX2(id) g_cx_reg_info2[id].type, KYTY_ID(id, 0), 0
#define KYTY_INDEX_SH2(id) g_sh_reg_info2[id].type, KYTY_ID(id, 1), 0
#define KYTY_INDEX_UC2(id) g_uc_reg_info2[id].type, KYTY_ID(id, 2), 0
#define KYTY_REG_CX1(id)   &g_cx_reg_info1[id].reg[0]
#define KYTY_REG_SH1(id)   &g_sh_reg_info1[id].reg[0]
#define KYTY_REG_UC1(id)   &g_uc_reg_info1[id].reg[0]
#define KYTY_REG_CX2(id)   &g_cx_reg_info2[id].reg[0]
#define KYTY_REG_SH2(id)   &g_sh_reg_info2[id].reg[0]
#define KYTY_REG_UC2(id)   &g_uc_reg_info2[id].reg[0]

static ShaderRegister* g_tbl_cx1[] = {
    KYTY_REG_CX1(0),  KYTY_REG_CX1(1),  KYTY_REG_CX1(2),  KYTY_REG_CX1(3),  KYTY_REG_CX1(4),  KYTY_REG_CX1(5),  KYTY_REG_CX1(6),
    KYTY_REG_CX1(7),  KYTY_REG_CX1(8),  KYTY_REG_CX1(9),  KYTY_REG_CX1(10), KYTY_REG_CX1(11), KYTY_REG_CX1(12), KYTY_REG_CX1(13),
    KYTY_REG_CX1(14), KYTY_REG_CX1(15), KYTY_REG_CX1(16), KYTY_REG_CX1(17), KYTY_REG_CX1(18), KYTY_REG_CX1(19), KYTY_REG_CX1(20),
    KYTY_REG_CX1(21), KYTY_REG_CX1(22), KYTY_REG_CX1(23), KYTY_REG_CX1(24), KYTY_REG_CX1(25), KYTY_REG_CX1(26), KYTY_REG_CX1(27),
    KYTY_REG_CX1(28), KYTY_REG_CX1(29), KYTY_REG_CX1(30), KYTY_REG_CX1(31), KYTY_REG_CX1(32), KYTY_REG_CX1(33), KYTY_REG_CX1(34),
    KYTY_REG_CX1(35), KYTY_REG_CX1(36), KYTY_REG_CX1(37), KYTY_REG_CX1(38), KYTY_REG_CX1(39), KYTY_REG_CX1(40), KYTY_REG_CX1(41),
    KYTY_REG_CX1(42), KYTY_REG_CX1(43), KYTY_REG_CX1(44), KYTY_REG_CX1(45), KYTY_REG_CX1(46), KYTY_REG_CX1(47), KYTY_REG_CX1(48),
    KYTY_REG_CX1(49), KYTY_REG_CX1(50), KYTY_REG_CX1(51), KYTY_REG_CX1(52), KYTY_REG_CX1(53), KYTY_REG_CX1(54), KYTY_REG_CX1(55),
    KYTY_REG_CX1(56), KYTY_REG_CX1(57), KYTY_REG_CX1(58), KYTY_REG_CX1(59), KYTY_REG_CX1(60), KYTY_REG_CX1(61), KYTY_REG_CX1(62),
    KYTY_REG_CX1(63), KYTY_REG_CX1(64), KYTY_REG_CX1(65), KYTY_REG_CX1(66), KYTY_REG_CX1(67), KYTY_REG_CX1(68), KYTY_REG_CX1(69),
    KYTY_REG_CX1(70), KYTY_REG_CX1(71), KYTY_REG_CX1(72), KYTY_REG_CX1(73), KYTY_REG_CX1(74), KYTY_REG_CX1(75), KYTY_REG_CX1(76),
    KYTY_REG_CX1(77)};

static ShaderRegister* g_tbl_sh1[]    = {KYTY_REG_SH1(0),  KYTY_REG_SH1(1),  KYTY_REG_SH1(2),  KYTY_REG_SH1(3),  KYTY_REG_SH1(4),
                                         KYTY_REG_SH1(5),  KYTY_REG_SH1(6),  KYTY_REG_SH1(7),  KYTY_REG_SH1(8),  KYTY_REG_SH1(9),
                                         KYTY_REG_SH1(10), KYTY_REG_SH1(11), KYTY_REG_SH1(12), KYTY_REG_SH1(13), KYTY_REG_SH1(14),
                                         KYTY_REG_SH1(15), KYTY_REG_SH1(16), KYTY_REG_SH1(17), KYTY_REG_SH1(18), KYTY_REG_SH1(19),
                                         KYTY_REG_SH1(20), KYTY_REG_SH1(21), KYTY_REG_SH1(22), KYTY_REG_SH1(23), KYTY_REG_SH1(24),
                                         KYTY_REG_SH1(25), KYTY_REG_SH1(26), KYTY_REG_SH1(27), KYTY_REG_SH1(28)};
static ShaderRegister* g_tbl_uc1[]    = {KYTY_REG_UC1(0),  KYTY_REG_UC1(1),  KYTY_REG_UC1(2),  KYTY_REG_UC1(3),  KYTY_REG_UC1(4),
                                         KYTY_REG_UC1(5),  KYTY_REG_UC1(6),  KYTY_REG_UC1(7),  KYTY_REG_UC1(8),  KYTY_REG_UC1(9),
                                         KYTY_REG_UC1(10), KYTY_REG_UC1(11), KYTY_REG_UC1(12), KYTY_REG_UC1(13), KYTY_REG_UC1(14),
                                         KYTY_REG_UC1(15), KYTY_REG_UC1(16), KYTY_REG_UC1(17), KYTY_REG_UC1(18), KYTY_REG_UC1(19)};
static uint32_t        g_tbl_index1[] = {
           KYTY_INDEX_CX1(0),  KYTY_INDEX_CX1(1),  KYTY_INDEX_CX1(2),  KYTY_INDEX_CX1(3),  KYTY_INDEX_CX1(4),  KYTY_INDEX_CX1(5),
           KYTY_INDEX_CX1(6),  KYTY_INDEX_CX1(7),  KYTY_INDEX_CX1(8),  KYTY_INDEX_CX1(9),  KYTY_INDEX_CX1(10), KYTY_INDEX_CX1(11),
           KYTY_INDEX_CX1(12), KYTY_INDEX_CX1(13), KYTY_INDEX_CX1(14), KYTY_INDEX_CX1(15), KYTY_INDEX_CX1(16), KYTY_INDEX_CX1(17),
           KYTY_INDEX_CX1(18), KYTY_INDEX_CX1(19), KYTY_INDEX_CX1(20), KYTY_INDEX_CX1(21), KYTY_INDEX_CX1(22), KYTY_INDEX_CX1(23),
           KYTY_INDEX_CX1(24), KYTY_INDEX_CX1(25), KYTY_INDEX_CX1(26), KYTY_INDEX_CX1(27), KYTY_INDEX_CX1(28), KYTY_INDEX_CX1(29),
           KYTY_INDEX_CX1(30), KYTY_INDEX_CX1(31), KYTY_INDEX_CX1(32), KYTY_INDEX_CX1(33), KYTY_INDEX_CX1(34), KYTY_INDEX_CX1(35),
           KYTY_INDEX_CX1(36), KYTY_INDEX_CX1(37), KYTY_INDEX_CX1(38), KYTY_INDEX_CX1(39), KYTY_INDEX_CX1(40), KYTY_INDEX_CX1(41),
           KYTY_INDEX_CX1(42), KYTY_INDEX_CX1(43), KYTY_INDEX_CX1(44), KYTY_INDEX_CX1(45), KYTY_INDEX_CX1(46), KYTY_INDEX_CX1(47),
           KYTY_INDEX_CX1(48), KYTY_INDEX_CX1(49), KYTY_INDEX_CX1(50), KYTY_INDEX_CX1(51), KYTY_INDEX_CX1(52), KYTY_INDEX_CX1(53),
           KYTY_INDEX_CX1(54), KYTY_INDEX_CX1(55), KYTY_INDEX_CX1(56), KYTY_INDEX_CX1(57), KYTY_INDEX_CX1(58), KYTY_INDEX_CX1(59),
           KYTY_INDEX_CX1(60), KYTY_INDEX_CX1(61), KYTY_INDEX_CX1(62), KYTY_INDEX_CX1(63), KYTY_INDEX_CX1(64), KYTY_INDEX_CX1(65),
           KYTY_INDEX_CX1(66), KYTY_INDEX_CX1(67), KYTY_INDEX_CX1(68), KYTY_INDEX_CX1(69), KYTY_INDEX_CX1(70), KYTY_INDEX_CX1(71),
           KYTY_INDEX_CX1(72), KYTY_INDEX_CX1(73), KYTY_INDEX_CX1(74), KYTY_INDEX_CX1(75), KYTY_INDEX_CX1(76), KYTY_INDEX_CX1(77),
           KYTY_INDEX_SH1(0),  KYTY_INDEX_SH1(1),  KYTY_INDEX_SH1(2),  KYTY_INDEX_SH1(3),  KYTY_INDEX_SH1(4),  KYTY_INDEX_SH1(5),
           KYTY_INDEX_SH1(6),  KYTY_INDEX_SH1(7),  KYTY_INDEX_SH1(8),  KYTY_INDEX_SH1(9),  KYTY_INDEX_SH1(10), KYTY_INDEX_SH1(11),
           KYTY_INDEX_SH1(12), KYTY_INDEX_SH1(13), KYTY_INDEX_SH1(14), KYTY_INDEX_SH1(15), KYTY_INDEX_SH1(16), KYTY_INDEX_SH1(17),
           KYTY_INDEX_SH1(18), KYTY_INDEX_SH1(19), KYTY_INDEX_SH1(20), KYTY_INDEX_SH1(21), KYTY_INDEX_SH1(22), KYTY_INDEX_SH1(23),
           KYTY_INDEX_SH1(24), KYTY_INDEX_SH1(25), KYTY_INDEX_SH1(26), KYTY_INDEX_SH1(27), KYTY_INDEX_SH1(28), KYTY_INDEX_UC1(0),
           KYTY_INDEX_UC1(1),  KYTY_INDEX_UC1(2),  KYTY_INDEX_UC1(3),  KYTY_INDEX_UC1(4),  KYTY_INDEX_UC1(5),  KYTY_INDEX_UC1(6),
           KYTY_INDEX_UC1(7),  KYTY_INDEX_UC1(8),  KYTY_INDEX_UC1(9),  KYTY_INDEX_UC1(10), KYTY_INDEX_UC1(11), KYTY_INDEX_UC1(12),
           KYTY_INDEX_UC1(13), KYTY_INDEX_UC1(14), KYTY_INDEX_UC1(15), KYTY_INDEX_UC1(16), KYTY_INDEX_UC1(17), KYTY_INDEX_UC1(18),
           KYTY_INDEX_UC1(19)};

static ShaderRegister* g_tbl_cx2[]    = {KYTY_REG_CX2(0), KYTY_REG_CX2(1), KYTY_REG_CX2(2), KYTY_REG_CX2(3)};
static ShaderRegister* g_tbl_sh2[]    = {KYTY_REG_SH2(0),  KYTY_REG_SH2(1),  KYTY_REG_SH2(2),  KYTY_REG_SH2(3),  KYTY_REG_SH2(4),
                                         KYTY_REG_SH2(5),  KYTY_REG_SH2(6),  KYTY_REG_SH2(7),  KYTY_REG_SH2(8),  KYTY_REG_SH2(9),
                                         KYTY_REG_SH2(10), KYTY_REG_SH2(11), KYTY_REG_SH2(12), KYTY_REG_SH2(13), KYTY_REG_SH2(14)};
static ShaderRegister* g_tbl_uc2[]    = {KYTY_REG_UC2(0), KYTY_REG_UC2(1), KYTY_REG_UC2(2)};
static uint32_t        g_tbl_index2[] = {KYTY_INDEX_CX2(0),  KYTY_INDEX_CX2(1),  KYTY_INDEX_CX2(2),  KYTY_INDEX_CX2(3),  KYTY_INDEX_SH2(0),
                                         KYTY_INDEX_SH2(1),  KYTY_INDEX_SH2(2),  KYTY_INDEX_SH2(3),  KYTY_INDEX_SH2(4),  KYTY_INDEX_SH2(5),
                                         KYTY_INDEX_SH2(6),  KYTY_INDEX_SH2(7),  KYTY_INDEX_SH2(8),  KYTY_INDEX_SH2(9),  KYTY_INDEX_SH2(10),
                                         KYTY_INDEX_SH2(11), KYTY_INDEX_SH2(12), KYTY_INDEX_SH2(13), KYTY_INDEX_SH2(14), KYTY_INDEX_UC2(0),
                                         KYTY_INDEX_UC2(1),  KYTY_INDEX_UC2(2)};

static RegisterDefaults g_reg_defaults1 = { // @suppress("Invalid arguments")
    g_tbl_cx1, g_tbl_sh1, g_tbl_uc1, nullptr, {0, 0}, g_tbl_index1, sizeof(g_tbl_index1) / 12};
static RegisterDefaults g_reg_defaults2 = { // @suppress("Invalid arguments")
    g_tbl_cx2, g_tbl_sh2, g_tbl_uc2, nullptr, {0, 0}, g_tbl_index2, sizeof(g_tbl_index2) / 12};


} } // namespace prosper::agc

// Returned by the sceAgcGetRegisterDefaults2 HLE thunk (ver is the SDK version, always 8).
extern "C" void* prosper_agc_reg_defaults(unsigned int /*ver*/) {
    return &prosper::agc::g_reg_defaults1;
}
extern "C" void* prosper_agc_reg_defaults_internal(unsigned int /*ver*/) {
    return &prosper::agc::g_reg_defaults2;
}
