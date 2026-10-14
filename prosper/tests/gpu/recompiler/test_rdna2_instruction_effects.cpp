// test_rdna2_instruction_effects — pins the shared RDNA2 instruction-effects table (#5146).
//
// The table replaces private opcode lists that had drifted apart: an SCC-reader list matched
// s_mov_b32 where it meant s_cmov_b32, a destination-width list wrote one SGPR for s_ashr_i64 and
// the WREXEC pair, and two copies of one width function disagreed about VOPC. Every one of those
// was silent -- a missed SCC reader drops the wave vote a consumer needs, and an under-counted
// destination lets an analysis believe a clobbered register still holds its old value.
//
// What is pinned here:
//   * COVERAGE: every opcode the decoder can produce in a tabulated format has an entry, named or
//     explicitly reserved. A new decodable opcode without an entry turns this red; so does deleting
//     an entry, because reserved slots are listed, not inferred from gaps.
//   * IDENTITY: the named opcodes are exactly the ones llvm-mc -mcpu=gfx1030 disassembles, under
//     the same mnemonic (an oracle independent of prosper; the arrays below were produced by
//     disassembling every encoding of each opcode space).
//   * EFFECTS: the SCC readers/writers and the SGPR-pair destinations against the ISA 70648
//     tables (chapter 5 "Sets SCC?" columns and the chapter 12 pseudo-code), listed independently.
//   * REGRESSIONS for the old-list bugs the table exposed, each against the old wrong answer.
#include "gpu/recompiler/rdna2_instruction_effects.hpp"

#include "gpu/recompiler/indirect/rdna2_indirect_scalar_writes.hpp"
#include "gpu/recompiler/rdna2_active_vector_words.hpp"
#include "gpu/recompiler/rdna2_cfg_registers.hpp"
#include "gpu/recompiler/rdna2_cfg_support.hpp"
#include "gpu/recompiler/rdna2_dead_wave_masks.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_scalar_condition_marks.hpp"
#include "gpu/recompiler/rdna2_recompile_shared.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include "gpu/recompiler/rdna2_vector_old_word.hpp"
#include "gpu/recompiler/rdna2_vector_word_marks.hpp"
#include "gpu/recompiler/rdna2_vgpr_word_definitions.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

using namespace prosper::gpu;

namespace {

struct LlvmOpcode {
    uint32_t opcode;
    const char* name;
};

// llvm-mc -arch=amdgcn -mcpu=gfx1030 -disassemble over each whole opcode space (SOPP with
// SIMM16 = 0, since several SOPP forms reject a nonzero immediate).
constexpr LlvmOpcode kLlvmSop2[] = {
    {0x00, "s_add_u32"},         {0x01, "s_sub_u32"},         {0x02, "s_add_i32"},
    {0x03, "s_sub_i32"},         {0x04, "s_addc_u32"},        {0x05, "s_subb_u32"},
    {0x06, "s_min_i32"},         {0x07, "s_min_u32"},         {0x08, "s_max_i32"},
    {0x09, "s_max_u32"},         {0x0a, "s_cselect_b32"},     {0x0b, "s_cselect_b64"},
    {0x0e, "s_and_b32"},         {0x0f, "s_and_b64"},         {0x10, "s_or_b32"},
    {0x11, "s_or_b64"},          {0x12, "s_xor_b32"},         {0x13, "s_xor_b64"},
    {0x14, "s_andn2_b32"},       {0x15, "s_andn2_b64"},       {0x16, "s_orn2_b32"},
    {0x17, "s_orn2_b64"},        {0x18, "s_nand_b32"},        {0x19, "s_nand_b64"},
    {0x1a, "s_nor_b32"},         {0x1b, "s_nor_b64"},         {0x1c, "s_xnor_b32"},
    {0x1d, "s_xnor_b64"},        {0x1e, "s_lshl_b32"},        {0x1f, "s_lshl_b64"},
    {0x20, "s_lshr_b32"},        {0x21, "s_lshr_b64"},        {0x22, "s_ashr_i32"},
    {0x23, "s_ashr_i64"},        {0x24, "s_bfm_b32"},         {0x25, "s_bfm_b64"},
    {0x26, "s_mul_i32"},         {0x27, "s_bfe_u32"},         {0x28, "s_bfe_i32"},
    {0x29, "s_bfe_u64"},         {0x2a, "s_bfe_i64"},         {0x2c, "s_absdiff_i32"},
    {0x2e, "s_lshl1_add_u32"},   {0x2f, "s_lshl2_add_u32"},   {0x30, "s_lshl3_add_u32"},
    {0x31, "s_lshl4_add_u32"},   {0x32, "s_pack_ll_b32_b16"}, {0x33, "s_pack_lh_b32_b16"},
    {0x34, "s_pack_hh_b32_b16"}, {0x35, "s_mul_hi_u32"},      {0x36, "s_mul_hi_i32"},
};
constexpr LlvmOpcode kLlvmSop1[] = {
    {0x03, "s_mov_b32"},
    {0x04, "s_mov_b64"},
    {0x05, "s_cmov_b32"},
    {0x06, "s_cmov_b64"},
    {0x07, "s_not_b32"},
    {0x08, "s_not_b64"},
    {0x09, "s_wqm_b32"},
    {0x0a, "s_wqm_b64"},
    {0x0b, "s_brev_b32"},
    {0x0c, "s_brev_b64"},
    {0x0d, "s_bcnt0_i32_b32"},
    {0x0e, "s_bcnt0_i32_b64"},
    {0x0f, "s_bcnt1_i32_b32"},
    {0x10, "s_bcnt1_i32_b64"},
    {0x11, "s_ff0_i32_b32"},
    {0x12, "s_ff0_i32_b64"},
    {0x13, "s_ff1_i32_b32"},
    {0x14, "s_ff1_i32_b64"},
    {0x15, "s_flbit_i32_b32"},
    {0x16, "s_flbit_i32_b64"},
    {0x17, "s_flbit_i32"},
    {0x18, "s_flbit_i32_i64"},
    {0x19, "s_sext_i32_i8"},
    {0x1a, "s_sext_i32_i16"},
    {0x1b, "s_bitset0_b32"},
    {0x1c, "s_bitset0_b64"},
    {0x1d, "s_bitset1_b32"},
    {0x1e, "s_bitset1_b64"},
    {0x1f, "s_getpc_b64"},
    {0x20, "s_setpc_b64"},
    {0x21, "s_swappc_b64"},
    {0x22, "s_rfe_b64"},
    {0x24, "s_and_saveexec_b64"},
    {0x25, "s_or_saveexec_b64"},
    {0x26, "s_xor_saveexec_b64"},
    {0x27, "s_andn2_saveexec_b64"},
    {0x28, "s_orn2_saveexec_b64"},
    {0x29, "s_nand_saveexec_b64"},
    {0x2a, "s_nor_saveexec_b64"},
    {0x2b, "s_xnor_saveexec_b64"},
    {0x2c, "s_quadmask_b32"},
    {0x2d, "s_quadmask_b64"},
    {0x2e, "s_movrels_b32"},
    {0x2f, "s_movrels_b64"},
    {0x30, "s_movreld_b32"},
    {0x31, "s_movreld_b64"},
    {0x34, "s_abs_i32"},
    {0x37, "s_andn1_saveexec_b64"},
    {0x38, "s_orn1_saveexec_b64"},
    {0x39, "s_andn1_wrexec_b64"},
    {0x3a, "s_andn2_wrexec_b64"},
    {0x3b, "s_bitreplicate_b64_b32"},
    {0x3c, "s_and_saveexec_b32"},
    {0x3d, "s_or_saveexec_b32"},
    {0x3e, "s_xor_saveexec_b32"},
    {0x3f, "s_andn2_saveexec_b32"},
    {0x40, "s_orn2_saveexec_b32"},
    {0x41, "s_nand_saveexec_b32"},
    {0x42, "s_nor_saveexec_b32"},
    {0x43, "s_xnor_saveexec_b32"},
    {0x44, "s_andn1_saveexec_b32"},
    {0x45, "s_orn1_saveexec_b32"},
    {0x46, "s_andn1_wrexec_b32"},
    {0x47, "s_andn2_wrexec_b32"},
    {0x49, "s_movrelsd_2_b32"},
};
constexpr LlvmOpcode kLlvmSopk[] = {
    {0x00, "s_movk_i32"},           {0x01, "s_version"},
    {0x02, "s_cmovk_i32"},          {0x03, "s_cmpk_eq_i32"},
    {0x04, "s_cmpk_lg_i32"},        {0x05, "s_cmpk_gt_i32"},
    {0x06, "s_cmpk_ge_i32"},        {0x07, "s_cmpk_lt_i32"},
    {0x08, "s_cmpk_le_i32"},        {0x09, "s_cmpk_eq_u32"},
    {0x0a, "s_cmpk_lg_u32"},        {0x0b, "s_cmpk_gt_u32"},
    {0x0c, "s_cmpk_ge_u32"},        {0x0d, "s_cmpk_lt_u32"},
    {0x0e, "s_cmpk_le_u32"},        {0x0f, "s_addk_i32"},
    {0x10, "s_mulk_i32"},           {0x12, "s_getreg_b32"},
    {0x13, "s_setreg_b32"},         {0x15, "s_setreg_imm32_b32"},
    {0x16, "s_call_b64"},           {0x17, "s_waitcnt_vscnt"},
    {0x18, "s_waitcnt_vmcnt"},      {0x19, "s_waitcnt_expcnt"},
    {0x1a, "s_waitcnt_lgkmcnt"},    {0x1b, "s_subvector_loop_begin"},
    {0x1c, "s_subvector_loop_end"},
};
constexpr LlvmOpcode kLlvmSopc[] = {
    {0x00, "s_cmp_eq_i32"},  {0x01, "s_cmp_lg_i32"},  {0x02, "s_cmp_gt_i32"},
    {0x03, "s_cmp_ge_i32"},  {0x04, "s_cmp_lt_i32"},  {0x05, "s_cmp_le_i32"},
    {0x06, "s_cmp_eq_u32"},  {0x07, "s_cmp_lg_u32"},  {0x08, "s_cmp_gt_u32"},
    {0x09, "s_cmp_ge_u32"},  {0x0a, "s_cmp_lt_u32"},  {0x0b, "s_cmp_le_u32"},
    {0x0c, "s_bitcmp0_b32"}, {0x0d, "s_bitcmp1_b32"}, {0x0e, "s_bitcmp0_b64"},
    {0x0f, "s_bitcmp1_b64"}, {0x12, "s_cmp_eq_u64"},  {0x13, "s_cmp_lg_u64"},
};
constexpr LlvmOpcode kLlvmSopp[] = {
    {0x00, "s_nop"},
    {0x01, "s_endpgm"},
    {0x02, "s_branch"},
    {0x03, "s_wakeup"},
    {0x04, "s_cbranch_scc0"},
    {0x05, "s_cbranch_scc1"},
    {0x06, "s_cbranch_vccz"},
    {0x07, "s_cbranch_vccnz"},
    {0x08, "s_cbranch_execz"},
    {0x09, "s_cbranch_execnz"},
    {0x0a, "s_barrier"},
    {0x0b, "s_setkill"},
    {0x0c, "s_waitcnt"},
    {0x0d, "s_sethalt"},
    {0x0e, "s_sleep"},
    {0x0f, "s_setprio"},
    {0x10, "s_sendmsg"},
    {0x11, "s_sendmsghalt"},
    {0x12, "s_trap"},
    {0x13, "s_icache_inv"},
    {0x14, "s_incperflevel"},
    {0x15, "s_decperflevel"},
    {0x16, "s_ttracedata"},
    {0x17, "s_cbranch_cdbgsys"},
    {0x18, "s_cbranch_cdbguser"},
    {0x19, "s_cbranch_cdbgsys_or_user"},
    {0x1a, "s_cbranch_cdbgsys_and_user"},
    {0x1b, "s_endpgm_saved"},
    {0x1e, "s_endpgm_ordered_ps_done"},
    {0x1f, "s_code_end"},
    {0x20, "s_inst_prefetch"},
    {0x21, "s_clause"},
    {0x22, "s_wait_idle"},
    {0x23, "s_waitcnt_depctr"},
    {0x24, "s_round_mode"},
    {0x25, "s_denorm_mode"},
    {0x28, "s_ttracedata_imm"},
};

// The VALU opcode spaces: the UNION of llvm-mc -mcpu=gfx1030 and -mcpu=gfx1013 (PS5 code uses
// gfx10.1-era encodings ISA 70648 dropped; see the table's comment), VOP3 without its VOPC band.
// Each encoding was disassembled with SGPR sources and again with no operands, so an instruction
// with no operands (v_nop, v_pipeflush, v_clrexcp) is named, not mistaken for a reserved slot.
// Where the two targets name one opcode differently (VOP2/VOP3 0x06, VOP3 0x140), gfx1030's name.
constexpr LlvmOpcode kLlvmVop2[] = {
    {0x001, "v_cndmask_b32"},    {0x002, "v_dot2c_f32_f16"},    {0x003, "v_add_f32"},
    {0x004, "v_sub_f32"},        {0x005, "v_subrev_f32"},       {0x006, "v_fmac_legacy_f32"},
    {0x007, "v_mul_legacy_f32"}, {0x008, "v_mul_f32"},          {0x009, "v_mul_i32_i24"},
    {0x00a, "v_mul_hi_i32_i24"}, {0x00b, "v_mul_u32_u24"},      {0x00c, "v_mul_hi_u32_u24"},
    {0x00d, "v_dot4c_i32_i8"},   {0x00f, "v_min_f32"},          {0x010, "v_max_f32"},
    {0x011, "v_min_i32"},        {0x012, "v_max_i32"},          {0x013, "v_min_u32"},
    {0x014, "v_max_u32"},        {0x016, "v_lshrrev_b32"},      {0x018, "v_ashrrev_i32"},
    {0x01a, "v_lshlrev_b32"},    {0x01b, "v_and_b32"},          {0x01c, "v_or_b32"},
    {0x01d, "v_xor_b32"},        {0x01e, "v_xnor_b32"},         {0x01f, "v_mac_f32"},
    {0x020, "v_madmk_f32"},      {0x021, "v_madak_f32"},        {0x025, "v_add_nc_u32"},
    {0x026, "v_sub_nc_u32"},     {0x027, "v_subrev_nc_u32"},    {0x028, "v_add_co_ci_u32"},
    {0x029, "v_sub_co_ci_u32"},  {0x02a, "v_subrev_co_ci_u32"}, {0x02b, "v_fmac_f32"},
    {0x02c, "v_fmamk_f32"},      {0x02d, "v_fmaak_f32"},        {0x02f, "v_cvt_pkrtz_f16_f32"},
    {0x032, "v_add_f16"},        {0x033, "v_sub_f16"},          {0x034, "v_subrev_f16"},
    {0x035, "v_mul_f16"},        {0x036, "v_fmac_f16"},         {0x037, "v_fmamk_f16"},
    {0x038, "v_fmaak_f16"},      {0x039, "v_max_f16"},          {0x03a, "v_min_f16"},
    {0x03b, "v_ldexp_f16"},      {0x03c, "v_pk_fmac_f16"},
};
constexpr LlvmOpcode kLlvmVop1[] = {
    {0x000, "v_nop"},
    {0x001, "v_mov_b32"},
    {0x002, "v_readfirstlane_b32"},
    {0x003, "v_cvt_i32_f64"},
    {0x004, "v_cvt_f64_i32"},
    {0x005, "v_cvt_f32_i32"},
    {0x006, "v_cvt_f32_u32"},
    {0x007, "v_cvt_u32_f32"},
    {0x008, "v_cvt_i32_f32"},
    {0x00a, "v_cvt_f16_f32"},
    {0x00b, "v_cvt_f32_f16"},
    {0x00c, "v_cvt_rpi_i32_f32"},
    {0x00d, "v_cvt_flr_i32_f32"},
    {0x00e, "v_cvt_off_f32_i4"},
    {0x00f, "v_cvt_f32_f64"},
    {0x010, "v_cvt_f64_f32"},
    {0x011, "v_cvt_f32_ubyte0"},
    {0x012, "v_cvt_f32_ubyte1"},
    {0x013, "v_cvt_f32_ubyte2"},
    {0x014, "v_cvt_f32_ubyte3"},
    {0x015, "v_cvt_u32_f64"},
    {0x016, "v_cvt_f64_u32"},
    {0x017, "v_trunc_f64"},
    {0x018, "v_ceil_f64"},
    {0x019, "v_rndne_f64"},
    {0x01a, "v_floor_f64"},
    {0x01b, "v_pipeflush"},
    {0x020, "v_fract_f32"},
    {0x021, "v_trunc_f32"},
    {0x022, "v_ceil_f32"},
    {0x023, "v_rndne_f32"},
    {0x024, "v_floor_f32"},
    {0x025, "v_exp_f32"},
    {0x027, "v_log_f32"},
    {0x02a, "v_rcp_f32"},
    {0x02b, "v_rcp_iflag_f32"},
    {0x02e, "v_rsq_f32"},
    {0x02f, "v_rcp_f64"},
    {0x031, "v_rsq_f64"},
    {0x033, "v_sqrt_f32"},
    {0x034, "v_sqrt_f64"},
    {0x035, "v_sin_f32"},
    {0x036, "v_cos_f32"},
    {0x037, "v_not_b32"},
    {0x038, "v_bfrev_b32"},
    {0x039, "v_ffbh_u32"},
    {0x03a, "v_ffbl_b32"},
    {0x03b, "v_ffbh_i32"},
    {0x03c, "v_frexp_exp_i32_f64"},
    {0x03d, "v_frexp_mant_f64"},
    {0x03e, "v_fract_f64"},
    {0x03f, "v_frexp_exp_i32_f32"},
    {0x040, "v_frexp_mant_f32"},
    {0x041, "v_clrexcp"},
    {0x042, "v_movreld_b32"},
    {0x043, "v_movrels_b32"},
    {0x044, "v_movrelsd_b32"},
    {0x048, "v_movrelsd_2_b32"},
    {0x050, "v_cvt_f16_u16"},
    {0x051, "v_cvt_f16_i16"},
    {0x052, "v_cvt_u16_f16"},
    {0x053, "v_cvt_i16_f16"},
    {0x054, "v_rcp_f16"},
    {0x055, "v_sqrt_f16"},
    {0x056, "v_rsq_f16"},
    {0x057, "v_log_f16"},
    {0x058, "v_exp_f16"},
    {0x059, "v_frexp_mant_f16"},
    {0x05a, "v_frexp_exp_i16_f16"},
    {0x05b, "v_floor_f16"},
    {0x05c, "v_ceil_f16"},
    {0x05d, "v_trunc_f16"},
    {0x05e, "v_rndne_f16"},
    {0x05f, "v_fract_f16"},
    {0x060, "v_sin_f16"},
    {0x061, "v_cos_f16"},
    {0x062, "v_sat_pk_u8_i16"},
    {0x063, "v_cvt_norm_i16_f16"},
    {0x064, "v_cvt_norm_u16_f16"},
    {0x065, "v_swap_b32"},
    {0x068, "v_swaprel_b32"},
};
constexpr LlvmOpcode kLlvmVopc[] = {
    {0x000, "v_cmp_f_f32"},     {0x001, "v_cmp_lt_f32"},     {0x002, "v_cmp_eq_f32"},
    {0x003, "v_cmp_le_f32"},    {0x004, "v_cmp_gt_f32"},     {0x005, "v_cmp_lg_f32"},
    {0x006, "v_cmp_ge_f32"},    {0x007, "v_cmp_o_f32"},      {0x008, "v_cmp_u_f32"},
    {0x009, "v_cmp_nge_f32"},   {0x00a, "v_cmp_nlg_f32"},    {0x00b, "v_cmp_ngt_f32"},
    {0x00c, "v_cmp_nle_f32"},   {0x00d, "v_cmp_neq_f32"},    {0x00e, "v_cmp_nlt_f32"},
    {0x00f, "v_cmp_tru_f32"},   {0x010, "v_cmpx_f_f32"},     {0x011, "v_cmpx_lt_f32"},
    {0x012, "v_cmpx_eq_f32"},   {0x013, "v_cmpx_le_f32"},    {0x014, "v_cmpx_gt_f32"},
    {0x015, "v_cmpx_lg_f32"},   {0x016, "v_cmpx_ge_f32"},    {0x017, "v_cmpx_o_f32"},
    {0x018, "v_cmpx_u_f32"},    {0x019, "v_cmpx_nge_f32"},   {0x01a, "v_cmpx_nlg_f32"},
    {0x01b, "v_cmpx_ngt_f32"},  {0x01c, "v_cmpx_nle_f32"},   {0x01d, "v_cmpx_neq_f32"},
    {0x01e, "v_cmpx_nlt_f32"},  {0x01f, "v_cmpx_tru_f32"},   {0x020, "v_cmp_f_f64"},
    {0x021, "v_cmp_lt_f64"},    {0x022, "v_cmp_eq_f64"},     {0x023, "v_cmp_le_f64"},
    {0x024, "v_cmp_gt_f64"},    {0x025, "v_cmp_lg_f64"},     {0x026, "v_cmp_ge_f64"},
    {0x027, "v_cmp_o_f64"},     {0x028, "v_cmp_u_f64"},      {0x029, "v_cmp_nge_f64"},
    {0x02a, "v_cmp_nlg_f64"},   {0x02b, "v_cmp_ngt_f64"},    {0x02c, "v_cmp_nle_f64"},
    {0x02d, "v_cmp_neq_f64"},   {0x02e, "v_cmp_nlt_f64"},    {0x02f, "v_cmp_tru_f64"},
    {0x030, "v_cmpx_f_f64"},    {0x031, "v_cmpx_lt_f64"},    {0x032, "v_cmpx_eq_f64"},
    {0x033, "v_cmpx_le_f64"},   {0x034, "v_cmpx_gt_f64"},    {0x035, "v_cmpx_lg_f64"},
    {0x036, "v_cmpx_ge_f64"},   {0x037, "v_cmpx_o_f64"},     {0x038, "v_cmpx_u_f64"},
    {0x039, "v_cmpx_nge_f64"},  {0x03a, "v_cmpx_nlg_f64"},   {0x03b, "v_cmpx_ngt_f64"},
    {0x03c, "v_cmpx_nle_f64"},  {0x03d, "v_cmpx_neq_f64"},   {0x03e, "v_cmpx_nlt_f64"},
    {0x03f, "v_cmpx_tru_f64"},  {0x080, "v_cmp_f_i32"},      {0x081, "v_cmp_lt_i32"},
    {0x082, "v_cmp_eq_i32"},    {0x083, "v_cmp_le_i32"},     {0x084, "v_cmp_gt_i32"},
    {0x085, "v_cmp_ne_i32"},    {0x086, "v_cmp_ge_i32"},     {0x087, "v_cmp_t_i32"},
    {0x088, "v_cmp_class_f32"}, {0x089, "v_cmp_lt_i16"},     {0x08a, "v_cmp_eq_i16"},
    {0x08b, "v_cmp_le_i16"},    {0x08c, "v_cmp_gt_i16"},     {0x08d, "v_cmp_ne_i16"},
    {0x08e, "v_cmp_ge_i16"},    {0x08f, "v_cmp_class_f16"},  {0x090, "v_cmpx_f_i32"},
    {0x091, "v_cmpx_lt_i32"},   {0x092, "v_cmpx_eq_i32"},    {0x093, "v_cmpx_le_i32"},
    {0x094, "v_cmpx_gt_i32"},   {0x095, "v_cmpx_ne_i32"},    {0x096, "v_cmpx_ge_i32"},
    {0x097, "v_cmpx_t_i32"},    {0x098, "v_cmpx_class_f32"}, {0x099, "v_cmpx_lt_i16"},
    {0x09a, "v_cmpx_eq_i16"},   {0x09b, "v_cmpx_le_i16"},    {0x09c, "v_cmpx_gt_i16"},
    {0x09d, "v_cmpx_ne_i16"},   {0x09e, "v_cmpx_ge_i16"},    {0x09f, "v_cmpx_class_f16"},
    {0x0a0, "v_cmp_f_i64"},     {0x0a1, "v_cmp_lt_i64"},     {0x0a2, "v_cmp_eq_i64"},
    {0x0a3, "v_cmp_le_i64"},    {0x0a4, "v_cmp_gt_i64"},     {0x0a5, "v_cmp_ne_i64"},
    {0x0a6, "v_cmp_ge_i64"},    {0x0a7, "v_cmp_t_i64"},      {0x0a8, "v_cmp_class_f64"},
    {0x0a9, "v_cmp_lt_u16"},    {0x0aa, "v_cmp_eq_u16"},     {0x0ab, "v_cmp_le_u16"},
    {0x0ac, "v_cmp_gt_u16"},    {0x0ad, "v_cmp_ne_u16"},     {0x0ae, "v_cmp_ge_u16"},
    {0x0b0, "v_cmpx_f_i64"},    {0x0b1, "v_cmpx_lt_i64"},    {0x0b2, "v_cmpx_eq_i64"},
    {0x0b3, "v_cmpx_le_i64"},   {0x0b4, "v_cmpx_gt_i64"},    {0x0b5, "v_cmpx_ne_i64"},
    {0x0b6, "v_cmpx_ge_i64"},   {0x0b7, "v_cmpx_t_i64"},     {0x0b8, "v_cmpx_class_f64"},
    {0x0b9, "v_cmpx_lt_u16"},   {0x0ba, "v_cmpx_eq_u16"},    {0x0bb, "v_cmpx_le_u16"},
    {0x0bc, "v_cmpx_gt_u16"},   {0x0bd, "v_cmpx_ne_u16"},    {0x0be, "v_cmpx_ge_u16"},
    {0x0c0, "v_cmp_f_u32"},     {0x0c1, "v_cmp_lt_u32"},     {0x0c2, "v_cmp_eq_u32"},
    {0x0c3, "v_cmp_le_u32"},    {0x0c4, "v_cmp_gt_u32"},     {0x0c5, "v_cmp_ne_u32"},
    {0x0c6, "v_cmp_ge_u32"},    {0x0c7, "v_cmp_t_u32"},      {0x0c8, "v_cmp_f_f16"},
    {0x0c9, "v_cmp_lt_f16"},    {0x0ca, "v_cmp_eq_f16"},     {0x0cb, "v_cmp_le_f16"},
    {0x0cc, "v_cmp_gt_f16"},    {0x0cd, "v_cmp_lg_f16"},     {0x0ce, "v_cmp_ge_f16"},
    {0x0cf, "v_cmp_o_f16"},     {0x0d0, "v_cmpx_f_u32"},     {0x0d1, "v_cmpx_lt_u32"},
    {0x0d2, "v_cmpx_eq_u32"},   {0x0d3, "v_cmpx_le_u32"},    {0x0d4, "v_cmpx_gt_u32"},
    {0x0d5, "v_cmpx_ne_u32"},   {0x0d6, "v_cmpx_ge_u32"},    {0x0d7, "v_cmpx_t_u32"},
    {0x0d8, "v_cmpx_f_f16"},    {0x0d9, "v_cmpx_lt_f16"},    {0x0da, "v_cmpx_eq_f16"},
    {0x0db, "v_cmpx_le_f16"},   {0x0dc, "v_cmpx_gt_f16"},    {0x0dd, "v_cmpx_lg_f16"},
    {0x0de, "v_cmpx_ge_f16"},   {0x0df, "v_cmpx_o_f16"},     {0x0e0, "v_cmp_f_u64"},
    {0x0e1, "v_cmp_lt_u64"},    {0x0e2, "v_cmp_eq_u64"},     {0x0e3, "v_cmp_le_u64"},
    {0x0e4, "v_cmp_gt_u64"},    {0x0e5, "v_cmp_ne_u64"},     {0x0e6, "v_cmp_ge_u64"},
    {0x0e7, "v_cmp_t_u64"},     {0x0e8, "v_cmp_u_f16"},      {0x0e9, "v_cmp_nge_f16"},
    {0x0ea, "v_cmp_nlg_f16"},   {0x0eb, "v_cmp_ngt_f16"},    {0x0ec, "v_cmp_nle_f16"},
    {0x0ed, "v_cmp_neq_f16"},   {0x0ee, "v_cmp_nlt_f16"},    {0x0ef, "v_cmp_tru_f16"},
    {0x0f0, "v_cmpx_f_u64"},    {0x0f1, "v_cmpx_lt_u64"},    {0x0f2, "v_cmpx_eq_u64"},
    {0x0f3, "v_cmpx_le_u64"},   {0x0f4, "v_cmpx_gt_u64"},    {0x0f5, "v_cmpx_ne_u64"},
    {0x0f6, "v_cmpx_ge_u64"},   {0x0f7, "v_cmpx_t_u64"},     {0x0f8, "v_cmpx_u_f16"},
    {0x0f9, "v_cmpx_nge_f16"},  {0x0fa, "v_cmpx_nlg_f16"},   {0x0fb, "v_cmpx_ngt_f16"},
    {0x0fc, "v_cmpx_nle_f16"},  {0x0fd, "v_cmpx_neq_f16"},   {0x0fe, "v_cmpx_nlt_f16"},
    {0x0ff, "v_cmpx_tru_f16"},
};
constexpr LlvmOpcode kLlvmVop3[] = {
    {0x101, "v_cndmask_b32"},
    {0x103, "v_add_f32"},
    {0x104, "v_sub_f32"},
    {0x105, "v_subrev_f32"},
    {0x106, "v_fmac_legacy_f32"},
    {0x107, "v_mul_legacy_f32"},
    {0x108, "v_mul_f32"},
    {0x109, "v_mul_i32_i24"},
    {0x10a, "v_mul_hi_i32_i24"},
    {0x10b, "v_mul_u32_u24"},
    {0x10c, "v_mul_hi_u32_u24"},
    {0x10f, "v_min_f32"},
    {0x110, "v_max_f32"},
    {0x111, "v_min_i32"},
    {0x112, "v_max_i32"},
    {0x113, "v_min_u32"},
    {0x114, "v_max_u32"},
    {0x116, "v_lshrrev_b32"},
    {0x118, "v_ashrrev_i32"},
    {0x11a, "v_lshlrev_b32"},
    {0x11b, "v_and_b32"},
    {0x11c, "v_or_b32"},
    {0x11d, "v_xor_b32"},
    {0x11e, "v_xnor_b32"},
    {0x11f, "v_mac_f32"},
    {0x125, "v_add_nc_u32"},
    {0x126, "v_sub_nc_u32"},
    {0x127, "v_subrev_nc_u32"},
    {0x128, "v_add_co_ci_u32"},
    {0x129, "v_sub_co_ci_u32"},
    {0x12a, "v_subrev_co_ci_u32"},
    {0x12b, "v_fmac_f32"},
    {0x12f, "v_cvt_pkrtz_f16_f32"},
    {0x132, "v_add_f16"},
    {0x133, "v_sub_f16"},
    {0x134, "v_subrev_f16"},
    {0x135, "v_mul_f16"},
    {0x136, "v_fmac_f16"},
    {0x139, "v_max_f16"},
    {0x13a, "v_min_f16"},
    {0x13b, "v_ldexp_f16"},
    {0x140, "v_fma_legacy_f32"},
    {0x141, "v_mad_f32"},
    {0x142, "v_mad_i32_i24"},
    {0x143, "v_mad_u32_u24"},
    {0x144, "v_cubeid_f32"},
    {0x145, "v_cubesc_f32"},
    {0x146, "v_cubetc_f32"},
    {0x147, "v_cubema_f32"},
    {0x148, "v_bfe_u32"},
    {0x149, "v_bfe_i32"},
    {0x14a, "v_bfi_b32"},
    {0x14b, "v_fma_f32"},
    {0x14c, "v_fma_f64"},
    {0x14d, "v_lerp_u8"},
    {0x14e, "v_alignbit_b32"},
    {0x14f, "v_alignbyte_b32"},
    {0x150, "v_mullit_f32"},
    {0x151, "v_min3_f32"},
    {0x152, "v_min3_i32"},
    {0x153, "v_min3_u32"},
    {0x154, "v_max3_f32"},
    {0x155, "v_max3_i32"},
    {0x156, "v_max3_u32"},
    {0x157, "v_med3_f32"},
    {0x158, "v_med3_i32"},
    {0x159, "v_med3_u32"},
    {0x15a, "v_sad_u8"},
    {0x15b, "v_sad_hi_u8"},
    {0x15c, "v_sad_u16"},
    {0x15d, "v_sad_u32"},
    {0x15e, "v_cvt_pk_u8_f32"},
    {0x15f, "v_div_fixup_f32"},
    {0x160, "v_div_fixup_f64"},
    {0x164, "v_add_f64"},
    {0x165, "v_mul_f64"},
    {0x166, "v_min_f64"},
    {0x167, "v_max_f64"},
    {0x168, "v_ldexp_f64"},
    {0x169, "v_mul_lo_u32"},
    {0x16a, "v_mul_hi_u32"},
    {0x16b, "v_mul_lo_i32"},
    {0x16c, "v_mul_hi_i32"},
    {0x16d, "v_div_scale_f32"},
    {0x16e, "v_div_scale_f64"},
    {0x16f, "v_div_fmas_f32"},
    {0x170, "v_div_fmas_f64"},
    {0x171, "v_msad_u8"},
    {0x172, "v_qsad_pk_u16_u8"},
    {0x173, "v_mqsad_pk_u16_u8"},
    {0x174, "v_trig_preop_f64"},
    {0x175, "v_mqsad_u32_u8"},
    {0x176, "v_mad_u64_u32"},
    {0x177, "v_mad_i64_i32"},
    {0x178, "v_xor3_b32"},
    {0x180, "v_nop"},
    {0x181, "v_mov_b32"},
    {0x183, "v_cvt_i32_f64"},
    {0x184, "v_cvt_f64_i32"},
    {0x185, "v_cvt_f32_i32"},
    {0x186, "v_cvt_f32_u32"},
    {0x187, "v_cvt_u32_f32"},
    {0x188, "v_cvt_i32_f32"},
    {0x18a, "v_cvt_f16_f32"},
    {0x18b, "v_cvt_f32_f16"},
    {0x18c, "v_cvt_rpi_i32_f32"},
    {0x18d, "v_cvt_flr_i32_f32"},
    {0x18e, "v_cvt_off_f32_i4"},
    {0x18f, "v_cvt_f32_f64"},
    {0x190, "v_cvt_f64_f32"},
    {0x191, "v_cvt_f32_ubyte0"},
    {0x192, "v_cvt_f32_ubyte1"},
    {0x193, "v_cvt_f32_ubyte2"},
    {0x194, "v_cvt_f32_ubyte3"},
    {0x195, "v_cvt_u32_f64"},
    {0x196, "v_cvt_f64_u32"},
    {0x197, "v_trunc_f64"},
    {0x198, "v_ceil_f64"},
    {0x199, "v_rndne_f64"},
    {0x19a, "v_floor_f64"},
    {0x19b, "v_pipeflush"},
    {0x1a0, "v_fract_f32"},
    {0x1a1, "v_trunc_f32"},
    {0x1a2, "v_ceil_f32"},
    {0x1a3, "v_rndne_f32"},
    {0x1a4, "v_floor_f32"},
    {0x1a5, "v_exp_f32"},
    {0x1a7, "v_log_f32"},
    {0x1aa, "v_rcp_f32"},
    {0x1ab, "v_rcp_iflag_f32"},
    {0x1ae, "v_rsq_f32"},
    {0x1af, "v_rcp_f64"},
    {0x1b1, "v_rsq_f64"},
    {0x1b3, "v_sqrt_f32"},
    {0x1b4, "v_sqrt_f64"},
    {0x1b5, "v_sin_f32"},
    {0x1b6, "v_cos_f32"},
    {0x1b7, "v_not_b32"},
    {0x1b8, "v_bfrev_b32"},
    {0x1b9, "v_ffbh_u32"},
    {0x1ba, "v_ffbl_b32"},
    {0x1bb, "v_ffbh_i32"},
    {0x1bc, "v_frexp_exp_i32_f64"},
    {0x1bd, "v_frexp_mant_f64"},
    {0x1be, "v_fract_f64"},
    {0x1bf, "v_frexp_exp_i32_f32"},
    {0x1c0, "v_frexp_mant_f32"},
    {0x1c1, "v_clrexcp"},
    {0x1c2, "v_movreld_b32"},
    {0x1c3, "v_movrels_b32"},
    {0x1c4, "v_movrelsd_b32"},
    {0x1c8, "v_movrelsd_2_b32"},
    {0x1d0, "v_cvt_f16_u16"},
    {0x1d1, "v_cvt_f16_i16"},
    {0x1d2, "v_cvt_u16_f16"},
    {0x1d3, "v_cvt_i16_f16"},
    {0x1d4, "v_rcp_f16"},
    {0x1d5, "v_sqrt_f16"},
    {0x1d6, "v_rsq_f16"},
    {0x1d7, "v_log_f16"},
    {0x1d8, "v_exp_f16"},
    {0x1d9, "v_frexp_mant_f16"},
    {0x1da, "v_frexp_exp_i16_f16"},
    {0x1db, "v_floor_f16"},
    {0x1dc, "v_ceil_f16"},
    {0x1dd, "v_trunc_f16"},
    {0x1de, "v_rndne_f16"},
    {0x1df, "v_fract_f16"},
    {0x1e0, "v_sin_f16"},
    {0x1e1, "v_cos_f16"},
    {0x1e2, "v_sat_pk_u8_i16"},
    {0x1e3, "v_cvt_norm_i16_f16"},
    {0x1e4, "v_cvt_norm_u16_f16"},
    {0x200, "v_interp_p1_f32"},
    {0x201, "v_interp_p2_f32"},
    {0x202, "v_interp_mov_f32"},
    {0x2ff, "v_lshlrev_b64"},
    {0x300, "v_lshrrev_b64"},
    {0x301, "v_ashrrev_i64"},
    {0x303, "v_add_nc_u16"},
    {0x304, "v_sub_nc_u16"},
    {0x305, "v_mul_lo_u16"},
    {0x307, "v_lshrrev_b16"},
    {0x308, "v_ashrrev_i16"},
    {0x309, "v_max_u16"},
    {0x30a, "v_max_i16"},
    {0x30b, "v_min_u16"},
    {0x30c, "v_min_i16"},
    {0x30d, "v_add_nc_i16"},
    {0x30e, "v_sub_nc_i16"},
    {0x30f, "v_add_co_u32"},
    {0x310, "v_sub_co_u32"},
    {0x311, "v_pack_b32_f16"},
    {0x312, "v_cvt_pknorm_i16_f16"},
    {0x313, "v_cvt_pknorm_u16_f16"},
    {0x314, "v_lshlrev_b16"},
    {0x319, "v_subrev_co_u32"},
    {0x340, "v_mad_u16"},
    {0x342, "v_interp_p1ll_f16"},
    {0x343, "v_interp_p1lv_f16"},
    {0x344, "v_perm_b32"},
    {0x345, "v_xad_u32"},
    {0x346, "v_lshl_add_u32"},
    {0x347, "v_add_lshl_u32"},
    {0x34b, "v_fma_f16"},
    {0x351, "v_min3_f16"},
    {0x352, "v_min3_i16"},
    {0x353, "v_min3_u16"},
    {0x354, "v_max3_f16"},
    {0x355, "v_max3_i16"},
    {0x356, "v_max3_u16"},
    {0x357, "v_med3_f16"},
    {0x358, "v_med3_i16"},
    {0x359, "v_med3_u16"},
    {0x35a, "v_interp_p2_f16"},
    {0x35e, "v_mad_i16"},
    {0x35f, "v_div_fixup_f16"},
    {0x360, "v_readlane_b32"},
    {0x361, "v_writelane_b32"},
    {0x362, "v_ldexp_f32"},
    {0x363, "v_bfm_b32"},
    {0x364, "v_bcnt_u32_b32"},
    {0x365, "v_mbcnt_lo_u32_b32"},
    {0x366, "v_mbcnt_hi_u32_b32"},
    {0x368, "v_cvt_pknorm_i16_f32"},
    {0x369, "v_cvt_pknorm_u16_f32"},
    {0x36a, "v_cvt_pk_u16_u32"},
    {0x36b, "v_cvt_pk_i16_i32"},
    {0x36d, "v_add3_u32"},
    {0x36f, "v_lshl_or_b32"},
    {0x371, "v_and_or_b32"},
    {0x372, "v_or3_b32"},
    {0x373, "v_mad_u32_u16"},
    {0x375, "v_mad_i32_i16"},
    {0x376, "v_sub_nc_i32"},
    {0x377, "v_permlane16_b32"},
    {0x378, "v_permlanex16_b32"},
    {0x37f, "v_add_nc_i32"},
};

const char* fmt_name(Rdna2Format fmt) {
    switch (fmt) {
        case Rdna2Format::SOP2: return "SOP2";
        case Rdna2Format::SOP1: return "SOP1";
        case Rdna2Format::SOPK: return "SOPK";
        case Rdna2Format::SOPC: return "SOPC";
        case Rdna2Format::SOPP: return "SOPP";
        case Rdna2Format::VOP1: return "VOP1";
        case Rdna2Format::VOP2: return "VOP2";
        case Rdna2Format::VOPC: return "VOPC";
        case Rdna2Format::VOP3: return "VOP3";
        default: return "other";
    }
}

// A one-dword instruction of `fmt` with opcode `op`, built from the encoding layout the decoder
// classifies on (rdna2_decode_one), plus a trailing literal in case the encoding wants one.
Rdna2Inst decode(Rdna2Format fmt, uint32_t op, uint32_t ssrc0 = 2u) {
    uint32_t code[2] = {0, 0x1234u};
    switch (fmt) {
        case Rdna2Format::SOP2:
            code[0] = 0x80000000u | (op << 23) | (4u << 16) | (6u << 8) | ssrc0;
            break;
        case Rdna2Format::SOP1: code[0] = 0xBE800000u | (4u << 16) | (op << 8) | ssrc0; break;
        case Rdna2Format::SOPK: code[0] = 0xB0000000u | (op << 23) | (4u << 16) | 0x10u; break;
        case Rdna2Format::SOPC: code[0] = 0xBF000000u | (op << 16) | (6u << 8) | ssrc0; break;
        case Rdna2Format::SOPP: code[0] = 0xBF800000u | (op << 16); break;
        // VALU forms with VGPR sources (v1, v2, v3), so no SDWA/DPP/literal field is triggered.
        case Rdna2Format::VOP1: code[0] = 0x7E000000u | (4u << 17) | (op << 9) | 257u; break;
        case Rdna2Format::VOP2: code[0] = (op << 25) | (4u << 17) | (2u << 9) | 257u; break;
        case Rdna2Format::VOPC: code[0] = 0x7C000000u | (op << 17) | (2u << 9) | 257u; break;
        case Rdna2Format::VOP3:
            code[0] = 0xD4000000u | (op << 16) | 4u;
            code[1] = 257u | (258u << 9) | (259u << 18);
            break;
        default: break;
    }
    return rdna2_decode_one(code, 2);
}

// A VOP3 instruction with SGPR sources s2, s6 and s10.
Rdna2Inst decode_vop3_sgpr(uint32_t op) {
    const uint32_t code[2] = {0xD4000000u | (op << 16) | 4u, 2u | (6u << 9) | (10u << 18)};
    return rdna2_decode_one(code, 2);
}

const Rdna2Format kTabulated[] = {Rdna2Format::SOP2, Rdna2Format::SOP1, Rdna2Format::SOPK,
                                  Rdna2Format::SOPC, Rdna2Format::SOPP};
const Rdna2Format kValu[] = {Rdna2Format::VOP2, Rdna2Format::VOP1, Rdna2Format::VOPC,
                             Rdna2Format::VOP3};
const Rdna2Format kEveryTabulated[] = {Rdna2Format::SOP2, Rdna2Format::SOP1, Rdna2Format::SOPK,
                                       Rdna2Format::SOPC, Rdna2Format::SOPP, Rdna2Format::VOP2,
                                       Rdna2Format::VOP1, Rdna2Format::VOPC, Rdna2Format::VOP3};

std::vector<LlvmOpcode> llvm_set(Rdna2Format fmt) {
    switch (fmt) {
        case Rdna2Format::SOP2: return {std::begin(kLlvmSop2), std::end(kLlvmSop2)};
        case Rdna2Format::SOP1: return {std::begin(kLlvmSop1), std::end(kLlvmSop1)};
        case Rdna2Format::SOPK: return {std::begin(kLlvmSopk), std::end(kLlvmSopk)};
        case Rdna2Format::SOPC: return {std::begin(kLlvmSopc), std::end(kLlvmSopc)};
        case Rdna2Format::SOPP: return {std::begin(kLlvmSopp), std::end(kLlvmSopp)};
        case Rdna2Format::VOP2: return {std::begin(kLlvmVop2), std::end(kLlvmVop2)};
        case Rdna2Format::VOP1: return {std::begin(kLlvmVop1), std::end(kLlvmVop1)};
        case Rdna2Format::VOPC: return {std::begin(kLlvmVopc), std::end(kLlvmVopc)};
        case Rdna2Format::VOP3: return {std::begin(kLlvmVop3), std::end(kLlvmVop3)};
        default: return {};
    }
}

bool in_set(std::initializer_list<uint32_t> set, uint32_t op) {
    for (uint32_t member : set)
        if (member == op) return true;
    return false;
}

}   // namespace

// ---- Coverage --------------------------------------------------------------------------------

TEST(Rdna2InstructionEffects, OpcodeSpaceIsExactlyWhatTheDecoderProduces) {
    for (Rdna2Format fmt : kEveryTabulated) {
        const Rdna2OpcodeRange range = rdna2_effects_opcode_range(fmt);
        ASSERT_LT(range.first, range.end) << fmt_name(fmt);
        for (uint32_t op = range.first; op < range.end; ++op) {
            const Rdna2Inst in = decode(fmt, op);
            EXPECT_EQ(in.fmt, fmt) << fmt_name(fmt) << " op=" << op;
            EXPECT_EQ(in.opcode, op) << fmt_name(fmt) << " op=" << op;
        }
    }
    // The first opcode past each space belongs to another format: the space is not too small.
    EXPECT_NE(decode(Rdna2Format::SOP2, 0x60).fmt, Rdna2Format::SOP2);
    EXPECT_NE(decode(Rdna2Format::SOPK, 0x1d).fmt, Rdna2Format::SOPK);
    EXPECT_NE(decode(Rdna2Format::VOP2, 0x3e).fmt, Rdna2Format::VOP2);
    EXPECT_NE(decode(Rdna2Format::VOP3, 0xff).fmt, Rdna2Format::VOP3);
}

TEST(Rdna2InstructionEffects, EveryDecodableOpcodeHasAnEntry) {
    for (Rdna2Format fmt : kEveryTabulated) {
        EXPECT_TRUE(rdna2_effects_format_is_tabulated(fmt)) << fmt_name(fmt);
        const Rdna2OpcodeRange range = rdna2_effects_opcode_range(fmt);
        for (uint32_t op = range.first; op < range.end; ++op)
            EXPECT_NE(rdna2_instruction_effects(fmt, op).status, Rdna2EffectsStatus::Unknown)
                << fmt_name(fmt) << " op=0x" << std::hex << op
                << " decodes but has no entry: add a named row or a reserved range";
    }
}

TEST(Rdna2InstructionEffects, RowsAreUniqueAndWellFormed) {
    std::set<std::pair<int, uint32_t>> seen;
    for (const Rdna2InstructionEffectsRow& row : rdna2_instruction_effects_rows()) {
        EXPECT_TRUE(seen.insert({static_cast<int>(row.fmt), row.opcode}).second)
            << fmt_name(row.fmt) << " op=0x" << std::hex << row.opcode << " has two rows";
        const Rdna2OpcodeRange range = rdna2_effects_opcode_range(row.fmt);
        EXPECT_TRUE(row.opcode >= range.first && row.opcode < range.end)
            << fmt_name(row.fmt) << " op=0x" << std::hex << row.opcode << " cannot be decoded";
        const Rdna2InstructionEffects& e = row.effects;
        if (e.status == Rdna2EffectsStatus::Known) {
            EXPECT_NE(e.name, nullptr);
            // The SCC read kind and the implicit-read bit are one fact stated twice.
            EXPECT_EQ(e.scc_read != Rdna2SccRead::None, (e.implicit_reads & kRdna2ImplicitScc) != 0)
                << e.name;
            EXPECT_EQ(e.exec != Rdna2ExecEffect::None,
                      (e.implicit_writes & kRdna2ImplicitExec) != 0)
                << e.name;
        } else if (e.status == Rdna2EffectsStatus::Valu) {
            EXPECT_NE(e.name, nullptr);
            EXPECT_TRUE(row.fmt == Rdna2Format::VOP1 || row.fmt == Rdna2Format::VOP2 ||
                        row.fmt == Rdna2Format::VOPC || row.fmt == Rdna2Format::VOP3);
            // The scalar-ALU fields are not described for a VALU row.
            EXPECT_EQ(e.sdst_dwords, 0u) << e.name;
            EXPECT_EQ(e.implicit_writes, 0u) << e.name;
            EXPECT_EQ(e.scc_read, Rdna2SccRead::None) << e.name;
            EXPECT_EQ(e.exec, Rdna2ExecEffect::None) << e.name;
            EXPECT_EQ(
                e.implicit_reads & ~(kRdna2ImplicitVcc | kRdna2ImplicitExec | kRdna2ImplicitM0), 0u)
                << e.name;
            // The vector-result fields state one fact each; these pairs must agree.
            EXPECT_EQ(e.m0_relative != Rdna2M0Relative::None,
                      e.lane_flow == Rdna2LaneFlow::Relative)
                << e.name;
            EXPECT_EQ((e.implicit_reads & kRdna2ImplicitM0) != 0,
                      e.lane_flow == Rdna2LaneFlow::Relative ||
                          e.lane_flow == Rdna2LaneFlow::Interpolate)
                << e.name;
            if (row.fmt == Rdna2Format::VOPC) {
                EXPECT_EQ(e.lane_flow, Rdna2LaneFlow::LaneWise) << e.name;
                EXPECT_EQ(e.vdst_dwords, 0u) << e.name << ": a compare writes a lane mask";
            }
            EXPECT_EQ(e.vdst_dwords == 0, row.fmt == Rdna2Format::VOPC ||
                                              e.lane_flow == Rdna2LaneFlow::None ||
                                              e.lane_flow == Rdna2LaneFlow::ReadLane)
                << e.name;
            if (e.accumulates) EXPECT_EQ(e.lane_flow, Rdna2LaneFlow::LaneWise) << e.name;
            if (e.writes_src0)
                EXPECT_TRUE(e.lane_flow == Rdna2LaneFlow::Swap ||
                            e.lane_flow == Rdna2LaneFlow::Relative)
                    << e.name;
        } else {
            EXPECT_EQ(e.status, Rdna2EffectsStatus::Reserved);
            EXPECT_EQ(e.name, nullptr);
        }
        // The lookup returns the row.
        EXPECT_EQ(rdna2_instruction_effects(row.fmt, row.opcode).status, e.status);
    }
}

TEST(Rdna2InstructionEffects, NamedOpcodesAreExactlyLlvmGfx1030) {
    for (Rdna2Format fmt : kEveryTabulated) {
        const std::vector<LlvmOpcode> llvm = llvm_set(fmt);
        ASSERT_FALSE(llvm.empty()) << fmt_name(fmt);
        std::set<uint32_t> llvm_ops;
        for (const LlvmOpcode& entry : llvm) {
            llvm_ops.insert(entry.opcode);
            const Rdna2InstructionEffects& e = rdna2_instruction_effects(fmt, entry.opcode);
            ASSERT_TRUE(e.known() || e.describes_vector_result())
                << fmt_name(fmt) << " op=0x" << std::hex << entry.opcode << " is " << entry.name
                << " in llvm-mc";
            EXPECT_STREQ(e.name, entry.name)
                << fmt_name(fmt) << " op=0x" << std::hex << entry.opcode;
        }
        const Rdna2OpcodeRange range = rdna2_effects_opcode_range(fmt);
        for (uint32_t op = range.first; op < range.end; ++op)
            if (!llvm_ops.contains(op))
                EXPECT_EQ(rdna2_instruction_effects(fmt, op).status, Rdna2EffectsStatus::Reserved)
                    << fmt_name(fmt) << " op=0x" << std::hex << op << " is invalid in llvm-mc";
    }
}

TEST(Rdna2InstructionEffects, UntabulatedLookupsAreUnknownNotNoEffect) {
    EXPECT_EQ(rdna2_instruction_effects(Rdna2Format::VOP3P, 0x10).status,
              Rdna2EffectsStatus::Unknown);
    // VOP3 0x000-0x0FF decode as VOPC, so they are not VOP3 entries.
    EXPECT_EQ(rdna2_instruction_effects(Rdna2Format::VOP3, 0x50).status,
              Rdna2EffectsStatus::Unknown);
    // A VALU entry describes its sources and vector result: it is not "known" for the scalar
    // questions (its SDST and implicit writes are not described).
    EXPECT_EQ(rdna2_instruction_effects(Rdna2Format::VOP3, 0x16d).status, Rdna2EffectsStatus::Valu);
    EXPECT_FALSE(rdna2_instruction_effects(Rdna2Format::VOP3, 0x16d).known());
    EXPECT_EQ(rdna2_instruction_effects(Rdna2Format::Unknown, 0).status,
              Rdna2EffectsStatus::Unknown);
    EXPECT_EQ(rdna2_instruction_effects(Rdna2Format::SOP1, 0x1000).status,
              Rdna2EffectsStatus::Unknown);
    EXPECT_EQ(rdna2_writes_scc(Rdna2Format::Unknown, 0), Rdna2Answer::Unknown);
    EXPECT_EQ(rdna2_reads_scc_implicitly(Rdna2Format::Unknown, 0), Rdna2Answer::Unknown);
    // A reserved scalar encoding is not a known non-writer either.
    EXPECT_EQ(rdna2_writes_scc(Rdna2Format::SOPC, 0x10), Rdna2Answer::Unknown);
    EXPECT_EQ(rdna2_reads_scc_implicitly(Rdna2Format::SOP2, 0x0c), Rdna2Answer::Unknown);
    // VALU, memory and export formats never touch SCC implicitly (ISA 3.8 / 3.9).
    for (Rdna2Format fmt :
         {Rdna2Format::VOP1, Rdna2Format::VOP2, Rdna2Format::VOPC, Rdna2Format::VOP3,
          Rdna2Format::VOP3P, Rdna2Format::SMEM, Rdna2Format::MIMG, Rdna2Format::EXP}) {
        EXPECT_EQ(rdna2_writes_scc(fmt, 0x10), Rdna2Answer::No);
        EXPECT_EQ(rdna2_reads_scc_implicitly(fmt, 0x10), Rdna2Answer::No);
    }
}

// ---- Effects against the ISA -------------------------------------------------------------------

TEST(Rdna2InstructionEffects, SccWritersMatchIsaTables) {
    // ISA 70648 5.3-5.8 "Sets SCC? y" and the 12.x pseudo-code (SCC = ...).
    const auto writers = [](Rdna2Format fmt, uint32_t op) {
        switch (fmt) {
            case Rdna2Format::SOP2:   // all but cselect, bfm, mul_i32, the packs and mul_hi
                return op <= 0x09 || (op >= 0x0e && op <= 0x23) || (op >= 0x27 && op <= 0x2a) ||
                       op == 0x2c || (op >= 0x2e && op <= 0x31);
            case Rdna2Format::SOP1:   // not, wqm, bcnt, quadmask, abs, every saveexec/wrexec
                return (op >= 0x07 && op <= 0x0a) || (op >= 0x0d && op <= 0x10) ||
                       (op >= 0x24 && op <= 0x2d) || op == 0x34 || (op >= 0x37 && op <= 0x3a) ||
                       (op >= 0x3c && op <= 0x47);
            case Rdna2Format::SOPK: return op >= 0x03 && op <= 0x0f;   // cmpk, addk
            case Rdna2Format::SOPC: return true;
            default: return false;
        }
    };
    for (Rdna2Format fmt : kTabulated)
        for (const LlvmOpcode& entry : llvm_set(fmt))
            EXPECT_EQ(rdna2_writes_scc(fmt, entry.opcode),
                      writers(fmt, entry.opcode) ? Rdna2Answer::Yes : Rdna2Answer::No)
                << entry.name;
}

TEST(Rdna2InstructionEffects, SccReadersMatchIsaTables) {
    // Data: s_addc/s_subb carry-in, s_cselect (ISA 5.4), s_cmov/s_cmovk (5.4). Branch: 12.5.
    for (Rdna2Format fmt : kTabulated) {
        for (const LlvmOpcode& entry : llvm_set(fmt)) {
            const uint32_t op = entry.opcode;
            const bool data = (fmt == Rdna2Format::SOP2 && in_set({0x04, 0x05, 0x0a, 0x0b}, op)) ||
                              (fmt == Rdna2Format::SOP1 && in_set({0x05, 0x06}, op)) ||
                              (fmt == Rdna2Format::SOPK && op == 0x02);
            const bool branch = fmt == Rdna2Format::SOPP && in_set({0x04, 0x05}, op);
            EXPECT_EQ(rdna2_instruction_effects(fmt, op).scc_read, data     ? Rdna2SccRead::Data
                                                                   : branch ? Rdna2SccRead::Branch
                                                                            : Rdna2SccRead::None)
                << entry.name;
        }
    }
}

TEST(Rdna2InstructionEffects, DestinationWidthsMatchIsaPseudoCode) {
    // D.u64 / D.i64 writers (ISA 12.x pseudo-code). Every other named SOP1/SOP2 writes one SGPR.
    const auto pair = [](Rdna2Format fmt, uint32_t op) {
        if (fmt == Rdna2Format::SOP2)
            return in_set({0x0b, 0x0f, 0x11, 0x13, 0x15, 0x17, 0x19, 0x1b, 0x1d, 0x1f, 0x21, 0x23,
                           0x25, 0x29, 0x2a},
                          op);
        if (fmt == Rdna2Format::SOP1)
            return in_set({0x04, 0x06, 0x08, 0x0a, 0x0c, 0x1c, 0x1e, 0x1f, 0x21, 0x2d, 0x2f, 0x31,
                           0x3b},
                          op) ||
                   (op >= 0x24 && op <= 0x2b) || (op >= 0x37 && op <= 0x3a);
        return fmt == Rdna2Format::SOPK && op == 0x16;   // s_call_b64
    };
    const auto none = [](Rdna2Format fmt, uint32_t op) {
        if (fmt == Rdna2Format::SOP1) return op == 0x20 || op == 0x22;   // s_setpc, s_rfe
        if (fmt ==
            Rdna2Format::SOPK)   // version, cmpk, setreg, waitcnt: SDST is a source or unused
            return op == 0x01 || (op >= 0x03 && op <= 0x0e) || op == 0x13 || op == 0x15 ||
                   (op >= 0x17 && op <= 0x1a);
        return fmt == Rdna2Format::SOPC || fmt == Rdna2Format::SOPP;
    };
    for (Rdna2Format fmt : kTabulated)
        for (const LlvmOpcode& entry : llvm_set(fmt))
            EXPECT_EQ(rdna2_instruction_effects(fmt, entry.opcode).sdst_dwords,
                      pair(fmt, entry.opcode)   ? 2u
                      : none(fmt, entry.opcode) ? 0u
                                                : 1u)
                << entry.name;
}

TEST(Rdna2InstructionEffects, ExecAndM0Effects) {
    for (uint32_t op = 0x24; op <= 0x2b; ++op)
        EXPECT_EQ(rdna2_instruction_effects(Rdna2Format::SOP1, op).exec, Rdna2ExecEffect::SaveExec);
    EXPECT_EQ(rdna2_instruction_effects(Rdna2Format::SOP1, 0x39).exec, Rdna2ExecEffect::WrExec);
    EXPECT_EQ(rdna2_instruction_effects(Rdna2Format::SOP1, 0x47).exec, Rdna2ExecEffect::WrExec);
    EXPECT_EQ(rdna2_instruction_effects(Rdna2Format::SOP1, 0x2e).m0_relative,
              Rdna2M0Relative::Source);
    EXPECT_EQ(rdna2_instruction_effects(Rdna2Format::SOP1, 0x30).m0_relative,
              Rdna2M0Relative::Destination);
    EXPECT_EQ(rdna2_instruction_effects(Rdna2Format::SOP1, 0x49).m0_relative,
              Rdna2M0Relative::Both);
    EXPECT_TRUE(rdna2_instruction_effects(Rdna2Format::SOPP, 0x06).implicit_reads &
                kRdna2ImplicitVcc);
    EXPECT_TRUE(rdna2_instruction_effects(Rdna2Format::SOPP, 0x08).implicit_reads &
                kRdna2ImplicitExec);
    // A SOPP never writes a register.
    for (const LlvmOpcode& entry : kLlvmSopp)
        EXPECT_EQ(rdna2_instruction_effects(Rdna2Format::SOPP, entry.opcode).implicit_writes, 0u)
            << entry.name;
}

// ---- Regressions: each of these was answered wrongly by a private list -------------------------

// The decoder retains a VGPR-shaped destination for these vector encodings, while emission
// writes the encoded scalar register. The indirect proofs must see that one-word clobber.
TEST(Rdna2InstructionEffectsRegression, ReadfirstlaneClobbersItsEncodedScalarDestination) {
    const uint32_t code[] = {0x7e000000u | (9u << 17) | (0x02u << 9) | 256u};
    Rdna2Inst in = rdna2_decode_one(code, std::size(code));
    ASSERT_EQ(in.fmt, Rdna2Format::VOP1);
    ASSERT_EQ(in.opcode, 0x02u);
    ASSERT_EQ(in.dst.value, 9);
    in.pc = 2;
    for (IndirectWave wave : {IndirectWave::Wave32, IndirectWave::Wave64}) {
        EXPECT_EQ(conservative_scalar_write_width(in, wave), 1u);
        EXPECT_FALSE(scalar_range_unchanged({in}, 1u, 3u, 9u, 1u, wave));
        EXPECT_TRUE(scalar_range_unchanged({in}, 1u, 3u, 8u, 1u, wave));
        EXPECT_TRUE(scalar_range_unchanged({in}, 1u, 3u, 10u, 1u, wave));
        EXPECT_TRUE(scalar_range_unchanged({in}, 2u, 3u, 9u, 1u, wave));
        EXPECT_TRUE(scalar_range_unchanged({in}, 1u, 2u, 9u, 1u, wave));
    }
    // Matched ordinary vector move writes v9, leaving s9 intact.
    const uint32_t mov_code[] = {0x7e000000u | (9u << 17) | (0x01u << 9) | 256u};
    Rdna2Inst mov = rdna2_decode_one(mov_code, std::size(mov_code));
    mov.pc = 2;
    EXPECT_TRUE(scalar_range_unchanged({mov}, 1u, 3u, 9u, 1u, IndirectWave::Wave64));
}

TEST(Rdna2InstructionEffectsRegression, ReadlaneClobbersItsEncodedScalarDestination) {
    // v_readlane_b32 s9, v0, 0: the selected lane comes from inline integer zero.
    const uint32_t code[] = {0xd7600009u, 256u | (128u << 9)};
    Rdna2Inst in = rdna2_decode_one(code, std::size(code));
    ASSERT_EQ(in.fmt, Rdna2Format::VOP3);
    ASSERT_EQ(in.opcode, 0x360u);
    ASSERT_EQ(in.dst.value, 9);
    in.pc = 2;
    for (IndirectWave wave : {IndirectWave::Wave32, IndirectWave::Wave64}) {
        EXPECT_EQ(conservative_scalar_write_width(in, wave), 1u);
        EXPECT_FALSE(scalar_range_unchanged({in}, 1u, 4u, 9u, 1u, wave));
        EXPECT_TRUE(scalar_range_unchanged({in}, 1u, 4u, 8u, 1u, wave));
        EXPECT_TRUE(scalar_range_unchanged({in}, 1u, 4u, 10u, 1u, wave));
    }
    // The adjacent WRITELANE encoding writes v9 and must not claim an s9 clobber.
    const uint32_t write_code[] = {0xd7610009u, 0u | (128u << 9)};
    Rdna2Inst write = rdna2_decode_one(write_code, std::size(write_code));
    write.pc = 2;
    EXPECT_TRUE(scalar_range_unchanged({write}, 1u, 4u, 9u, 1u, IndirectWave::Wave64));
}

// The two copies of conservative_scalar_write_width "disagreed" about a VOPC destination (1 SGPR
// in the descriptor-range proof, 2 in the pointer analysis). Both were right for the one wave size
// their own proof requires, so the shared copy takes the wave size; charging a Wave32 proof a pair
// refuses its exact shape. Both copies charged s_call_b64 one SGPR, but it writes a pair, and
// both charged a Wave64 VOP3B carry-out one SGPR, but it is a pair too.
TEST(Rdna2InstructionEffectsRegression, ConservativeWidthChargesEveryPairDestination) {
    Rdna2Inst compare;
    compare.fmt = Rdna2Format::VOPC;
    compare.opcode = 0xc1;   // v_cmp_lt_u32, e64 form writing s[8:9] (Wave64) or s8 (Wave32)
    compare.dst = {OperandKind::SGPR, 8};
    EXPECT_EQ(conservative_scalar_write_width(compare, IndirectWave::Wave64), 2u);
    EXPECT_EQ(conservative_scalar_write_width(compare, IndirectWave::Wave32), 1u);
    const Rdna2Inst call = decode(Rdna2Format::SOPK, 0x16);
    EXPECT_EQ(conservative_scalar_write_width(call, IndirectWave::Wave32), 2u)
        << "s_call_b64 writes a pair";
    const Rdna2Inst movk = decode(Rdna2Format::SOPK, 0x00);
    EXPECT_EQ(conservative_scalar_write_width(movk, IndirectWave::Wave64), 1u);
    // So a Wave64 compare writing s[8:9] clobbers a live range starting at s9; in Wave32 it does not.
    compare.pc = 2;
    EXPECT_FALSE(scalar_range_unchanged({compare}, 1u, 3u, 9u, 1u, IndirectWave::Wave64));
    EXPECT_TRUE(scalar_range_unchanged({compare}, 1u, 3u, 9u, 1u, IndirectWave::Wave32));
    // A Wave64 v_add_co_u32's carry-out s[4:5] clobbers s5.
    Rdna2Inst carry;
    carry.fmt = Rdna2Format::VOP3;
    carry.opcode = kVop3OpcodeAddCoU32;
    carry.pc = 2;
    carry.dst = {OperandKind::VGPR, 0};
    carry.sdst = {OperandKind::SGPR, 4};
    EXPECT_FALSE(scalar_range_unchanged({carry}, 1u, 3u, 5u, 1u, IndirectWave::Wave64));
    EXPECT_TRUE(scalar_range_unchanged({carry}, 1u, 3u, 5u, 1u, IndirectWave::Wave32));
    // v_readfirstlane_b32 s9, v0: the decoder files s9 as a VGPR-kind operand, and both old
    // copies charged it nothing, so a live range at s9 looked untouched.
    const uint32_t readfirstlane[] = {0x7E000000u | (9u << 17) | (0x02u << 9) | 256u};
    Rdna2Inst lane = rdna2_decode_one(readfirstlane, 1);
    ASSERT_EQ(lane.fmt, Rdna2Format::VOP1);
    ASSERT_EQ(lane.dst.kind, OperandKind::VGPR);
    lane.pc = 2;
    EXPECT_EQ(conservative_scalar_write_width(lane, IndirectWave::Wave32), 1u);
    EXPECT_FALSE(scalar_range_unchanged({lane}, 1u, 3u, 9u, 1u, IndirectWave::Wave32));
    EXPECT_EQ(scalar_write_width(lane), 1u);
}

// rdna2_recompile_fragment.cpp's shader_reads_scc listed SOP1 0x02/0x03 as s_cmov. 0x03 is
// s_mov_b32 and 0x02 is reserved; s_cmov_b32/b64 are 0x05/0x06. Nearly every shader contains an
// s_mov_b32, so the scan said "reads SCC" for almost everything (paying a wave vote it did not need)
// and missed a shader whose only reader was an s_cmov. It also ignored an explicit SCC operand.
TEST(Rdna2InstructionEffectsRegression, ProgramReadsSccRecognisesCmovNotMov) {
    EXPECT_FALSE(rdna2_program_reads_scc({decode(Rdna2Format::SOP1, kSop1OpcodeMovB32)}))
        << "s_mov_b32 does not read SCC (the old scan said it did)";
    EXPECT_TRUE(rdna2_program_reads_scc({decode(Rdna2Format::SOP1, kSop1OpcodeCmovB32)}))
        << "s_cmov_b32 reads SCC (the old scan missed it)";
    EXPECT_TRUE(rdna2_program_reads_scc({decode(Rdna2Format::SOP1, kSop1OpcodeCmovB64)}));
    EXPECT_TRUE(rdna2_program_reads_scc({decode(Rdna2Format::SOP1, kSop1OpcodeMovB32, 253u)}))
        << "s_mov_b32 s4, scc reads SCC through its operand";
    EXPECT_TRUE(rdna2_program_reads_scc({decode(Rdna2Format::SOPP, kSoppOpcodeCbranchScc0)}));
    EXPECT_FALSE(rdna2_program_reads_scc({decode(Rdna2Format::SOPP, kSoppOpcodeCbranchExecz)}));
}

// dead_wave_mask_writes elides a dead `s_and_b64 vcc, ...` only when nothing reads SCC, because the
// elided instruction also produces SCC. Its private reader list ignored an explicit SCC operand, so
// a program reading SCC through `s_mov_b32 sN, scc` lost the SCC its mask op computed.
TEST(Rdna2InstructionEffectsRegression, DeadWaveMaskElisionSeesAnSccOperandReader) {
    // s_and_b64 vcc, s[0:1], vcc ; s_mov_b32 s2, scc ; s_endpgm
    const uint32_t code[] = {0x80000000u | (0x0fu << 23) | (106u << 16) | (106u << 8) | 0u,
                             0xBE800000u | (2u << 16) | (kSop1OpcodeMovB32 << 8) | 253u,
                             0xBF810000u};
    std::vector<Rdna2Inst> ins;
    rdna2_walk(code, std::size(code), ins);
    ASSERT_EQ(ins.size(), 3u);
    ASSERT_EQ(ins[0].fmt, Rdna2Format::SOP2);
    ASSERT_EQ(ins[1].src[0].value, 253);
    EXPECT_TRUE(dead_wave_mask_writes(ins).empty())
        << "the mask op's SCC is read, so it must not be elided";
    // Positive control: without the SCC read the same mask op is dead and elided.
    std::vector<Rdna2Inst> no_reader = {ins[0], ins[2]};
    no_reader[1].pc = ins[0].pc + ins[0].len_dwords;
    EXPECT_EQ(dead_wave_mask_writes(no_reader).count(ins[0].pc), 1u);
}

// scalar_write_width answered one SGPR for these 64-bit destinations (and for s_rfe_b64, which
// writes none), so liveness and provenance believed the high word survived.
TEST(Rdna2InstructionEffectsRegression, ScalarWriteWidthCountsEveryPairDestination) {
    struct Case {
        Rdna2Format fmt;
        uint32_t op;
        uint32_t width;
        const char* name;
    };
    const Case cases[] = {
        {Rdna2Format::SOP2, 0x23, 2, "s_ashr_i64"},
        {Rdna2Format::SOP2, 0x2a, 2, "s_bfe_i64"},
        {Rdna2Format::SOP1, 0x0c, 2, "s_brev_b64"},
        {Rdna2Format::SOP1, 0x1c, 2, "s_bitset0_b64"},
        {Rdna2Format::SOP1, 0x1e, 2, "s_bitset1_b64"},
        {Rdna2Format::SOP1, 0x21, 2, "s_swappc_b64"},
        {Rdna2Format::SOP1, 0x31, 2, "s_movreld_b64"},
        {Rdna2Format::SOP1, 0x39, 2, "s_andn1_wrexec_b64"},
        {Rdna2Format::SOP1, 0x3a, 2, "s_andn2_wrexec_b64"},
        {Rdna2Format::SOP1, 0x22, 0, "s_rfe_b64"},
        {Rdna2Format::SOPK, 0x16, 2, "s_call_b64"},
        {Rdna2Format::SOPK, 0x1b, 1, "s_subvector_loop_begin"},
    };
    for (const Case& c : cases) {
        const Rdna2Inst in = decode(c.fmt, c.op);
        EXPECT_EQ(scalar_write_width(in), c.width) << c.name;
    }
    EXPECT_TRUE(sopk_writes_scalar_data(0x16)) << "s_call_b64 writes its return address";
}

// rdna2_sop2_dest_dwords reported a pair for every B32 opcode past 0x28, so an analysis believed
// s_lshl2_add_u32 s4 also clobbered s5. scalar_write_width said one dword for the same opcodes.
TEST(Rdna2InstructionEffectsRegression, Sop2DestDwordsAgreesWithScalarWriteWidth) {
    for (const LlvmOpcode& entry : kLlvmSop2)
        EXPECT_EQ(rdna2_sop2_dest_dwords(entry.opcode),
                  scalar_write_width(decode(Rdna2Format::SOP2, entry.opcode)))
            << entry.name;
    EXPECT_EQ(rdna2_sop2_dest_dwords(0x2f), 1u) << "s_lshl2_add_u32";
    EXPECT_EQ(rdna2_sop2_dest_dwords(0x0c), 2u) << "a reserved opcode fails closed to a pair";
}

// scalar_instruction_preserves_scc answered "unknown" for SOP1 forms the ISA says leave SCC alone
// (Table 14: "Sets SCC? n"), which cost the Wave32 SCC provenance a refresh it was entitled to.
TEST(Rdna2InstructionEffectsRegression, SccPreservationCoversEveryNonWritingSop1) {
    for (uint32_t op : {0x0cu, 0x11u, 0x12u, 0x17u, 0x18u, 0x19u, 0x1au, 0x1cu, 0x1eu, 0x2fu, 0x30u,
                        0x31u, 0x49u}) {
        const Rdna2Inst in = decode(Rdna2Format::SOP1, op);
        EXPECT_TRUE(scalar_instruction_preserves_scc(in)) << "SOP1 op=0x" << std::hex << op;
        EXPECT_FALSE(scalar_instruction_writes_scc(in)) << "SOP1 op=0x" << std::hex << op;
    }
    // A reserved encoding is neither.
    const Rdna2Inst reserved = decode(Rdna2Format::SOPC, 0x10);
    EXPECT_FALSE(scalar_instruction_preserves_scc(reserved));
    EXPECT_FALSE(scalar_instruction_writes_scc(reserved));
}

// scalar_alu_source_words derived most scalar source widths from the DESTINATION width, so every
// instruction whose source is wider than its result was charged one SGPR for a pair read: the
// B64 bit counts and finds, the B64 bit compares, and (through the width bugs above) every B64
// form scalar_write_width undercounted. A pair read charged one word lets a proof treat the high
// word as unread.
TEST(Rdna2InstructionEffectsRegression, ScalarSourceWidthsCountPairReads) {
    struct Case {
        Rdna2Format fmt;
        uint32_t op;
        uint32_t source;
        uint32_t width;
        const char* name;
    };
    const Case cases[] = {
        {Rdna2Format::SOP1, 0x0e, 0, 2, "s_bcnt0_i32_b64"},
        {Rdna2Format::SOP1, 0x12, 0, 2, "s_ff0_i32_b64"},
        {Rdna2Format::SOP1, 0x0c, 0, 2, "s_brev_b64"},
        {Rdna2Format::SOP1, 0x39, 0, 2, "s_andn1_wrexec_b64"},
        {Rdna2Format::SOPC, 0x0f, 0, 2, "s_bitcmp1_b64"},
        {Rdna2Format::SOPC, 0x0f, 1, 1, "s_bitcmp1_b64 bit index"},
        {Rdna2Format::SOP2, 0x23, 0, 2, "s_ashr_i64"},
        {Rdna2Format::SOP2, 0x23, 1, 1, "s_ashr_i64 shift count"},
        {Rdna2Format::SOP1, 0x1c, 0, 1, "s_bitset0_b64 bit index"},
        {Rdna2Format::SOP1, 0x10, 0, 2, "s_bcnt1_i32_b64 (was already right)"},
    };
    for (const Case& c : cases)
        EXPECT_EQ(scalar_alu_source_words(decode(c.fmt, c.op), c.source), c.width) << c.name;
}

// ---- VALU source widths ------------------------------------------------------------------------

// An ISA-side structural check, independent of the llvm-mc derivation the rows came from: a pair
// source appears exactly where the mnemonic names a 64-bit operand type, apart from a short list
// of operands the ISA types by other means (lane masks, the SAD forms' bit ranges).
TEST(Rdna2InstructionEffectsValu, PairSourcesFollowTheMnemonic) {
    const auto names_64bit = [](const std::string& name) {
        for (const char* suffix : {"_f64", "_i64", "_u64", "_b64"})
            if (name.find(suffix) != std::string::npos) return true;
        return false;
    };
    // name -> source index read as a pair although the name carries no 64-bit type.
    const std::set<std::pair<std::string, uint32_t>> untyped_pairs = {
        {"v_cndmask_b32", 2},      {"v_add_co_ci_u32", 2},   {"v_sub_co_ci_u32", 2},
        {"v_subrev_co_ci_u32", 2}, {"v_qsad_pk_u16_u8", 0},  {"v_qsad_pk_u16_u8", 2},
        {"v_mqsad_pk_u16_u8", 0},  {"v_mqsad_pk_u16_u8", 2}, {"v_mqsad_u32_u8", 0},
        {"v_mqsad_u32_u8", 2},
    };
    size_t rows = 0;
    for (const Rdna2InstructionEffectsRow& row : rdna2_instruction_effects_rows()) {
        if (row.effects.status != Rdna2EffectsStatus::Valu) continue;
        ++rows;
        const std::string name = row.effects.name;
        bool any_pair = false;
        for (uint32_t source = 0; source < 3; ++source) {
            const uint32_t width = row.effects.ssrc_dwords[source];
            EXPECT_LE(width, 2u) << name;
            if (width != 2) continue;
            any_pair = true;
            EXPECT_TRUE(names_64bit(name) || untyped_pairs.contains({name, source}))
                << name << " source " << source << " is a pair but its name has no 64-bit type";
        }
        if (names_64bit(name) && name.find("cvt_") == std::string::npos &&
            name.find("lshl") == std::string::npos)
            EXPECT_TRUE(any_pair) << name << " names a 64-bit type but reads no pair";
    }
    EXPECT_GT(rows, 500u);
}

// scalar_alu_source_words charged every VOP3 opcode missing from a hand-kept B32 allowlist TWO
// SGPRs per source, so a B32 read of s9 also demanded s10 and a fresh-mask guard refused valid
// shaders (#5148 added V_BCNT_U32_B32 one opcode at a time). It also charged one SGPR for the pair
// read of F64 VOP1/VOP2/VOPC operands. Each case below was answered wrongly by that code.
TEST(Rdna2InstructionEffectsRegression, ValuSourceWidthsComeFromTheIsaNotAnAllowlist) {
    struct Case {
        Rdna2Format fmt;
        uint32_t op;
        uint32_t source;
        uint32_t width;
        const char* name;
    };
    const Case cases[] = {
        // The comment's "known remainder" of B32 forms left on the pair default.
        {Rdna2Format::VOP3, 0x345, 0, 1, "v_xad_u32"},
        {Rdna2Format::VOP3, 0x363, 0, 1, "v_bfm_b32"},
        {Rdna2Format::VOP3, 0x372, 2, 1, "v_or3_b32"},
        {Rdna2Format::VOP3, 0x178, 1, 1, "v_xor3_b32"},
        {Rdna2Format::VOP3, 0x149, 0, 1, "v_bfe_i32"},
        {Rdna2Format::VOP3, 0x158, 0, 1, "v_med3_i32"},
        {Rdna2Format::VOP3, 0x103, 0, 1, "v_add_f32_e64"},
        // Pair reads the old code charged one SGPR.
        {Rdna2Format::VOP3, 0x18f, 0, 2, "v_cvt_f32_f64_e64"},
        {Rdna2Format::VOP3, 0x164, 1, 2, "v_add_f64"},
        // Unchanged answers the table must keep.
        {Rdna2Format::VOP3, 0x364, 0, 1, "v_bcnt_u32_b32"},
        {Rdna2Format::VOP3, 0x101, 2, 2, "v_cndmask_b32_e64 lane mask"},
        {Rdna2Format::VOP3, 0x176, 2, 2, "v_mad_u64_u32 addend"},
        {Rdna2Format::VOP3, 0x2ff, 0, 1, "v_lshlrev_b64 shift"},
        {Rdna2Format::VOP3, 0x2ff, 1, 2, "v_lshlrev_b64 value"},
    };
    for (const Case& c : cases)
        EXPECT_EQ(scalar_alu_source_words(decode_vop3_sgpr(c.op), c.source), c.width) << c.name;
    // A source the instruction does not have: the decoder fills SRC2 of every VOP3.
    EXPECT_EQ(scalar_alu_source_words(decode_vop3_sgpr(0x103), 2), UINT32_MAX) << "v_add_f32_e64";
    // VOPC: an F64 compare and its e32 form read pairs; CLASS_F64's class mask is one SGPR.
    const uint32_t cmp_lt_f64_e64[2] = {0xD4000000u | (0x21u << 16) | 106u, 2u | (6u << 9)};
    const Rdna2Inst cmp = rdna2_decode_one(cmp_lt_f64_e64, 2);
    ASSERT_EQ(cmp.fmt, Rdna2Format::VOPC);
    EXPECT_EQ(scalar_alu_source_words(cmp, 0), 2u) << "v_cmp_lt_f64_e64";
    EXPECT_EQ(scalar_alu_source_words(cmp, 1), 2u) << "v_cmp_lt_f64_e64";
    const uint32_t class_f64_e64[2] = {0xD4000000u | (0xa8u << 16) | 106u, 2u | (6u << 9)};
    const Rdna2Inst cls = rdna2_decode_one(class_f64_e64, 2);
    EXPECT_EQ(scalar_alu_source_words(cls, 0), 2u) << "v_cmp_class_f64";
    EXPECT_EQ(scalar_alu_source_words(cls, 1), 1u) << "v_cmp_class_f64 class mask";
    // VOP1 e32 with an SGPR source: v_cvt_f32_f64 v4, s[2:3].
    const uint32_t cvt_e32[1] = {0x7E000000u | (4u << 17) | (0x0fu << 9) | 2u};
    EXPECT_EQ(scalar_alu_source_words(rdna2_decode_one(cvt_e32, 1), 0), 2u) << "v_cvt_f32_f64_e32";
}

// ---- VALU vector results -----------------------------------------------------------------------

// ISA-side checks of the vector-result fields, each against a list or rule stated here
// independently of the generator that wrote the rows (which read VDST's register class from
// llvm-mc): the 16-bit results are the ISA 70648 rows whose pseudo-code assigns D.f16, D.f16_lo,
// D.u16, D.i16 or D.u/i[15:0]; the pair results are the mnemonics whose first type (AMD names the
// destination type first) is 64-bit; the accumulators are the rows whose pseudo-code reads D.
TEST(Rdna2InstructionEffectsValu, VectorResultFactsMatchTheIsa) {
    const std::set<std::string> half_results = {
        "v_cvt_f16_f32",      "v_cvt_f16_u16", "v_cvt_f16_i16",    "v_cvt_u16_f16",
        "v_cvt_i16_f16",      "v_rcp_f16",     "v_sqrt_f16",       "v_rsq_f16",
        "v_log_f16",          "v_exp_f16",     "v_frexp_mant_f16", "v_frexp_exp_i16_f16",
        "v_floor_f16",        "v_ceil_f16",    "v_trunc_f16",      "v_rndne_f16",
        "v_fract_f16",        "v_sin_f16",     "v_cos_f16",        "v_cvt_norm_i16_f16",
        "v_cvt_norm_u16_f16", "v_add_f16",     "v_sub_f16",        "v_subrev_f16",
        "v_mul_f16",          "v_fmac_f16",    "v_fmamk_f16",      "v_fmaak_f16",
        "v_max_f16",          "v_min_f16",     "v_ldexp_f16",      "v_add_nc_u16",
        "v_sub_nc_u16",       "v_mul_lo_u16",  "v_lshrrev_b16",    "v_ashrrev_i16",
        "v_max_u16",          "v_max_i16",     "v_min_u16",        "v_min_i16",
        "v_add_nc_i16",       "v_sub_nc_i16",  "v_lshlrev_b16",    "v_mad_u16",
        "v_fma_f16",          "v_min3_f16",    "v_min3_i16",       "v_min3_u16",
        "v_max3_f16",         "v_max3_i16",    "v_max3_u16",       "v_med3_f16",
        "v_med3_i16",         "v_med3_u16",    "v_interp_p2_f16",  "v_mad_i16",
        "v_div_fixup_f16",
    };
    const std::set<std::string> accumulators = {
        "v_mac_f32",     "v_fmac_f32",      "v_fmac_legacy_f32", "v_fmac_f16",
        "v_pk_fmac_f16", "v_dot2c_f32_f16", "v_dot4c_i32_i8"};
    const std::map<std::string, Rdna2LaneFlow> flows = {
        {"v_nop", Rdna2LaneFlow::None},
        {"v_pipeflush", Rdna2LaneFlow::None},
        {"v_clrexcp", Rdna2LaneFlow::None},
        {"v_readfirstlane_b32", Rdna2LaneFlow::ReadLane},
        {"v_readlane_b32", Rdna2LaneFlow::ReadLane},
        {"v_writelane_b32", Rdna2LaneFlow::WriteLane},
        {"v_permlane16_b32", Rdna2LaneFlow::CrossLane},
        {"v_permlanex16_b32", Rdna2LaneFlow::CrossLane},
        {"v_mbcnt_lo_u32_b32", Rdna2LaneFlow::LaneIndex},
        {"v_mbcnt_hi_u32_b32", Rdna2LaneFlow::LaneIndex},
        {"v_swap_b32", Rdna2LaneFlow::Swap},
        {"v_movreld_b32", Rdna2LaneFlow::Relative},
        {"v_movrels_b32", Rdna2LaneFlow::Relative},
        {"v_movrelsd_b32", Rdna2LaneFlow::Relative},
        {"v_movrelsd_2_b32", Rdna2LaneFlow::Relative},
        {"v_swaprel_b32", Rdna2LaneFlow::Relative},
        {"v_interp_p1_f32", Rdna2LaneFlow::Interpolate},
        {"v_interp_p2_f32", Rdna2LaneFlow::Interpolate},
        {"v_interp_mov_f32", Rdna2LaneFlow::Interpolate},
        {"v_interp_p1ll_f16", Rdna2LaneFlow::Interpolate},
        {"v_interp_p1lv_f16", Rdna2LaneFlow::Interpolate},
        {"v_interp_p2_f16", Rdna2LaneFlow::Interpolate},
    };
    // VDST is not a VGPR for these; for the rest the first type token names D's width, except
    // the SAD forms whose packed or 128-bit results the mnemonic does not type.
    const std::map<std::string, uint32_t> width_exceptions = {
        {"v_readfirstlane_b32", 0u}, {"v_readlane_b32", 0u}, {"v_nop", 0u},
        {"v_pipeflush", 0u},         {"v_clrexcp", 0u},      {"v_qsad_pk_u16_u8", 2u},
        {"v_mqsad_pk_u16_u8", 2u},   {"v_mqsad_u32_u8", 4u},
    };
    const auto first_type_bits = [](const std::string& name) -> uint32_t {
        size_t at = name.find('_');
        while (at != std::string::npos) {
            const size_t next = name.find('_', at + 1);
            const std::string token =
                name.substr(at + 1, next == std::string::npos ? std::string::npos : next - at - 1);
            if (token.size() >= 2 && std::strchr("fuib", token[0]) &&
                token.find_first_not_of("0123456789", 1) == std::string::npos)
                return static_cast<uint32_t>(std::stoul(token.substr(1)));
            at = next;
        }
        return 32;
    };
    size_t rows = 0, halves = 0, pairs = 0;
    for (const Rdna2InstructionEffectsRow& row : rdna2_instruction_effects_rows()) {
        const Rdna2InstructionEffects& e = row.effects;
        if (!e.describes_vector_result()) continue;
        ++rows;
        const std::string name = e.name;
        if (row.fmt == Rdna2Format::VOPC) continue;   // lane-mask results; pinned in RowsAre...
        EXPECT_EQ(e.vdst_half, half_results.contains(name)) << name;
        EXPECT_EQ(e.accumulates, accumulators.contains(name)) << name;
        const auto flow = flows.find(name);
        EXPECT_EQ(e.lane_flow, flow == flows.end() ? Rdna2LaneFlow::LaneWise : flow->second)
            << name;
        const auto width = width_exceptions.find(name);
        const uint32_t expected = width != width_exceptions.end() ? width->second
                                  : first_type_bits(name) == 64   ? 2u
                                                                  : 1u;
        EXPECT_EQ(e.vdst_dwords, expected) << name;
        halves += e.vdst_half;
        pairs += e.vdst_dwords == 2;
    }
    EXPECT_GT(rows, 550u);
    EXPECT_EQ(halves, 86u);   // every encoding of every 16-bit result
    EXPECT_GT(pairs, 30u);
    // Implicit reads: VCC only where no SRC2 names the mask, M0 for the relative/interp forms,
    // EXEC for v_readfirstlane (it reads the first ACTIVE lane).
    EXPECT_EQ(rdna2_instruction_effects(Rdna2Format::VOP2, 0x01).implicit_reads, kRdna2ImplicitVcc);
    EXPECT_EQ(rdna2_instruction_effects(Rdna2Format::VOP3, 0x101).implicit_reads, 0u)
        << "v_cndmask_b32_e64 names its mask as SRC2";
    EXPECT_EQ(rdna2_instruction_effects(Rdna2Format::VOP3, 0x16f).implicit_reads, kRdna2ImplicitVcc)
        << "v_div_fmas_f32";
    EXPECT_EQ(rdna2_instruction_effects(Rdna2Format::VOP1, 0x02).implicit_reads,
              kRdna2ImplicitExec);
    EXPECT_TRUE(rdna2_instruction_effects(Rdna2Format::VOP1, 0x65).writes_src0) << "v_swap_b32";
}

// The question helpers combine the opcode's facts with the encoding's modifiers.
TEST(Rdna2InstructionEffectsValu, CompleteResultsHonourTheEncoding) {
    Rdna2Inst add3 = decode(Rdna2Format::VOP3, kVop3OpcodeAdd3U32);
    EXPECT_EQ(rdna2_valu_complete_result_dwords(add3), 1u);
    EXPECT_EQ(rdna2_valu_lane_mapped(Rdna2Format::VOP3, kVop3OpcodeAdd3U32), Rdna2Answer::Yes);
    // A 16-bit result defines its word only through SDWA DWORD + UNUSED_PAD.
    Rdna2Inst add_f16 = decode(Rdna2Format::VOP2, 0x32);
    EXPECT_EQ(rdna2_valu_complete_result_dwords(add_f16), 0u);
    add_f16.has_sdwa = true;
    add_f16.sdwa_dst_sel = 6;
    add_f16.sdwa_dst_unused = 0;
    EXPECT_EQ(rdna2_valu_complete_result_dwords(add_f16), 1u);
    add_f16.sdwa_dst_unused = 2;
    EXPECT_EQ(rdna2_valu_complete_result_dwords(add_f16), 0u) << "UNUSED_PRESERVE";
    // DPP keeps masked rows/banks.
    Rdna2Inst mov = decode(Rdna2Format::VOP1, 0x01);
    EXPECT_EQ(rdna2_valu_complete_result_dwords(mov), 1u);
    mov.has_dpp = true;
    EXPECT_EQ(rdna2_valu_complete_result_dwords(mov), 0u);
    // An accumulator's result is not one of its explicit operands alone.
    EXPECT_EQ(rdna2_valu_complete_result_dwords(decode(Rdna2Format::VOP2, 0x2b)), 0u);
    EXPECT_EQ(rdna2_valu_result_keeps_old_vdst(Rdna2Format::VOP2, 0x2b), Rdna2Answer::Yes);
    // A reserved slot answers Unknown, and the complete-result count fails closed to 0.
    EXPECT_EQ(rdna2_valu_lane_mapped(Rdna2Format::VOP3, 0x13c), Rdna2Answer::Unknown);
    EXPECT_EQ(rdna2_valu_complete_result_dwords(decode(Rdna2Format::VOP3, 0x13c)), 0u);
    EXPECT_EQ(rdna2_valu_lane_mapped(Rdna2Format::SOP2, 0), Rdna2Answer::Unknown);
}

// Each case below was answered wrongly by a private opcode list the table replaced (#5143,
// #5146 step 4). The old answer is in the comment; every assertion fails against it.
TEST(Rdna2InstructionEffectsRegression, ValuResultQuestionsComeFromTheTable) {
    // v_add3_u32 and every other lane-wise VOP3-only operation: the active-word lists stopped at
    // VOP3 0x300 with one hand-added exception (v_lshl_add_u32), so a genuine ADD3 address never
    // became current DATA (#5143). Old: false.
    EXPECT_TRUE(vector_write_completes_active_word(decode(Rdna2Format::VOP3, kVop3OpcodeAdd3U32)));
    EXPECT_TRUE(vector_write_completes_active_word(decode(Rdna2Format::VOP3, 0x372)))
        << "v_or3_b32";
    EXPECT_TRUE(vector_write_completes_active_word(decode(Rdna2Format::VOP3, 0x36f)))
        << "v_lshl_or_b32";
    // v_frexp_mant_f32 sat inside a VOP1 "0x40-0x47" exclusion meant for v_movrel*. Old: false.
    EXPECT_TRUE(vector_write_completes_active_word(decode(Rdna2Format::VOP1, 0x40)));
    // M0-relative moves outside that range were counted lane-wise and complete. Old: true.
    EXPECT_FALSE(vector_write_completes_active_word(decode(Rdna2Format::VOP1, 0x48)))
        << "v_movrelsd_2_b32";
    EXPECT_FALSE(vector_write_completes_active_word(decode(Rdna2Format::VOP1, 0x68)))
        << "v_swaprel_b32";
    EXPECT_FALSE(vector_write_completes_active_word(decode(Rdna2Format::VOP3, 0x1c3)))
        << "v_movrels_b32_e64";
    EXPECT_FALSE(vector_write_completes_active_word(decode(Rdna2Format::VOP3, 0x200)))
        << "v_interp_p1_f32_e64 reads LDS";
    // Accumulators and 16-bit results the old-word list did not name. Old: false.
    for (const auto& [fmt, op, name] :
         std::initializer_list<std::tuple<Rdna2Format, uint32_t, const char*>>{
             {Rdna2Format::VOP2, 0x02, "v_dot2c_f32_f16"},
             {Rdna2Format::VOP2, 0x06, "v_fmac_legacy_f32"},
             {Rdna2Format::VOP2, 0x0d, "v_dot4c_i32_i8"},
             {Rdna2Format::VOP2, 0x36, "v_fmac_f16"},
             {Rdna2Format::VOP3, 0x106, "v_fmac_legacy_f32_e64"},
             {Rdna2Format::VOP3, 0x136, "v_fmac_f16_e64"},
             {Rdna2Format::VOP3, 0x132, "v_add_f16_e64"},
             {Rdna2Format::VOP3, 0x18a, "v_cvt_f16_f32_e64"},
             {Rdna2Format::VOP2, 0x3b, "v_ldexp_f16"},
             {Rdna2Format::VOP3, 0x1c2, "v_movreld_b32_e64"},
         })
        EXPECT_TRUE(vector_write_depends_on_old_word(decode(fmt, op), true)) << name;
    // VGPR destination widths: every F64 result is a pair; v_nop writes nothing. Old: 1.
    EXPECT_EQ(rdna2_vgpr_write_count(decode(Rdna2Format::VOP3, 0x164)), 2u) << "v_add_f64";
    EXPECT_EQ(rdna2_vgpr_write_count(decode(Rdna2Format::VOP1, 0x10)), 2u) << "v_cvt_f64_f32";
    EXPECT_EQ(rdna2_vgpr_write_count(decode(Rdna2Format::VOP3, 0x301)), 2u) << "v_ashrrev_i64";
    EXPECT_EQ(rdna2_vgpr_write_count(decode(Rdna2Format::VOP1, 0x00)), 0u) << "v_nop";
    // Unchanged answers the table must keep.
    EXPECT_EQ(rdna2_vgpr_write_count(decode(Rdna2Format::VOP3, 0x176)), 2u) << "v_mad_u64_u32";
    EXPECT_EQ(rdna2_vgpr_write_count(decode(Rdna2Format::VOP1, 0x02)), 0u) << "v_readfirstlane";
    EXPECT_TRUE(vector_write_depends_on_old_word(decode(Rdna2Format::VOP3, 0x361), true))
        << "v_writelane_b32";
    EXPECT_TRUE(vector_write_depends_on_old_word(decode(Rdna2Format::VOP3, 0x13c), true))
        << "0x13c is not an architectural e64 form; it keeps the old word conservatively";
    // Complete-word results the hand list omitted (old: 0), and an F64 result's pair.
    EXPECT_EQ(ordinary_vgpr_complete_results(decode(Rdna2Format::VOP3, 0x11b)), 1u)
        << "v_and_b32_e64";
    EXPECT_EQ(ordinary_vgpr_complete_results(decode(Rdna2Format::VOP3, 0x16b)), 1u)
        << "v_mul_lo_i32";
    EXPECT_EQ(ordinary_vgpr_complete_results(decode(Rdna2Format::VOP3, 0x164)), 2u) << "v_add_f64";
    EXPECT_EQ(ordinary_vgpr_complete_results(decode(Rdna2Format::VOP3, 0x11f)), 0u)
        << "v_mac_f32_e64 reads the old VDST";
    // Storage inventories: an M0-relative destination in any encoding may reach every VGPR above
    // the encoded one, and a swap also writes SRC0. Old: only VOP1 v_movreld_b32; no SRC0 write.
    const Rdna2Inst movreld_e64 = decode(Rdna2Format::VOP3, 0x1c2);
    EXPECT_TRUE(dynamic_vgpr_destination(movreld_e64));
    std::vector<int> written;
    for_each_possible_vector_write(movreld_e64, 7, [&](int reg) { written.push_back(reg); });
    EXPECT_EQ(written, (std::vector<int>{4, 5, 6, 7}));
    const Rdna2Inst swap = decode(Rdna2Format::VOP1, 0x65);   // v_swap_b32 v4, v1
    written.clear();
    for_each_possible_vector_write(swap, 255, [&](int reg) { written.push_back(reg); });
    EXPECT_EQ(written, (std::vector<int>{4, 1}));
    EXPECT_TRUE(writes_vgpr(swap, 1));
}

// active_vector_sources_defined decides whether a VALU result is current DATA. Accumulators need
// their old VDST in every encoding (old: only v_mac_f32; v_fmac_f32 was refused outright and the
// v_dot*c / legacy / f16 forms were admitted without it), and implicit VCC comes from the table.
TEST(Rdna2InstructionEffectsRegression, ActiveVectorSourcesFollowTheTable) {
    ActiveVectorWords with_dst, without_dst;
    for (int reg = 0; reg < 8; ++reg) {
        with_dst.current.insert(reg);
        if (reg != 4) without_dst.current.insert(reg);
    }
    std::set<int> relayed;
    for (int reg = 0; reg < 8; ++reg) relayed.insert(reg);
    const std::set<int> vcc = {106}, none;
    const auto defined = [&](const ActiveVectorWords& state, Rdna2Format fmt, uint32_t op,
                             const std::set<int>& masks) {
        return active_vector_sources_defined(state, decode(fmt, op), relayed, masks, {}, {}, true);
    };
    EXPECT_TRUE(defined(with_dst, Rdna2Format::VOP3, kVop3OpcodeAdd3U32, none)) << "old: false";
    EXPECT_TRUE(defined(with_dst, Rdna2Format::VOP2, 0x2b, none)) << "v_fmac_f32; old: false";
    EXPECT_FALSE(defined(without_dst, Rdna2Format::VOP2, 0x2b, none)) << "v_fmac_f32";
    EXPECT_FALSE(defined(without_dst, Rdna2Format::VOP2, 0x02, none)) << "v_dot2c; old: true";
    EXPECT_FALSE(defined(without_dst, Rdna2Format::VOP3, 0x12b, none)) << "fmac_e64; old: true";
    EXPECT_FALSE(defined(with_dst, Rdna2Format::VOP1, 0x48, none)) << "movrelsd_2; old: true";
    EXPECT_FALSE(defined(with_dst, Rdna2Format::VOP3, 0x16f, none)) << "div_fmas; old: true";
    EXPECT_TRUE(defined(with_dst, Rdna2Format::VOP3, 0x16f, vcc)) << "v_div_fmas_f32";
    // Unchanged: v_mac_f32 needs its VDST; v_cndmask_b32 (VOP2) needs VCC.
    EXPECT_FALSE(defined(without_dst, Rdna2Format::VOP2, 0x1f, none));
    EXPECT_TRUE(defined(with_dst, Rdna2Format::VOP2, 0x1f, none));
    EXPECT_FALSE(defined(with_dst, Rdna2Format::VOP2, 0x01, none));
    EXPECT_TRUE(defined(with_dst, Rdna2Format::VOP2, 0x01, vcc));
}
