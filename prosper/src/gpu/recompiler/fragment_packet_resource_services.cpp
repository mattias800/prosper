#include "gpu/recompiler/fragment_packet_services.hpp"
#include "gpu/recompiler/rdna2_alu_support.hpp"
#include <bit>

namespace prosper::gpu {
namespace {
uint32_t input_word(SpirvCompute& b, uint32_t index) {
    const auto pointer = b.id(), value = b.id();
    b.put(b.code, Op_AccessChain, {b.t_ptr_sb_f32, pointer, b.v_in, b.uconst(0), index});
    b.put(b.code, Op_Load, {b.t_u32, value, pointer});
    return value;
}
uint32_t and_(SpirvCompute& b, uint32_t a, uint32_t c) { return b.ucmp(Op_LogicalAnd, a, c); }
uint32_t normal_or_zero(SpirvCompute& b, uint32_t bits) {
    const auto mag = b.ibin(Op_BitwiseAnd, bits, b.uconst(0x7fffffffu));
    return b.ucmp(Op_LogicalOr, b.ucmp(Op_IEqual, mag, b.uconst(0)),
        and_(b, b.ucmp(Op_UGreaterThanEqual, mag, b.uconst(0x800000u)),
                b.ucmp(Op_ULessThan, mag, b.uconst(0x7f800000u))));
}
uint32_t exact_center(SpirvCompute& b, uint32_t bits, uint32_t extent) {
    // For W=2^n and 0<u<1, u is (2*x+1)/(2*W) iff its full significand has
    // r=149-n-E low zero bits followed by an odd bit. Descriptor dimensions fit 14 bits;
    // the range guard therefore gives 9<=r<=23. Sanitize before EVERY emitted shift so
    // even a failing operand executes defined integer code while all workers rendezvous.
    const uint32_t n = std::countr_zero(extent);
    const auto e = b.ibin(Op_BitwiseAnd, b.ibin(Op_ShiftRightLogical, bits, b.uconst(23)), b.uconst(255));
    const auto range = and_(b, b.ucmp(Op_UGreaterThanEqual, e, b.uconst(126 - n)),
        b.ucmp(Op_ULessThanEqual, e, b.uconst(126)));
    const auto r = b.sel(range, b.ibin(Op_ISub, b.uconst(149 - n), e), b.uconst(0));
    const auto bit = b.ibin(Op_ShiftLeftLogical, b.uconst(1), r);
    const auto sig = b.ibin(Op_BitwiseOr, b.ibin(Op_BitwiseAnd, bits, b.uconst(0x7fffff)), b.uconst(0x800000));
    return and_(b, and_(b, range, b.ucmp(Op_IEqual,
        b.ibin(Op_BitwiseAnd, bits, b.uconst(0x80000000)), b.uconst(0))),
        and_(b, b.ucmp(Op_IEqual, b.ibin(Op_BitwiseAnd, sig,
            b.ibin(Op_ISub, bit, b.uconst(1))), b.uconst(0)),
            b.ucmp(Op_INotEqual, b.ibin(Op_BitwiseAnd, sig, bit), b.uconst(0))));
}
void write_vector(SpirvCompute& b, RegState& state, int reg, uint32_t bits) {
    // The packet inventory REQUIRED the genuine old value, including inactive lanes. A service
    // cannot manufacture backing for an EXEC-off write or a subsequent EXEC-ignoring READLANE.
    state.vreg[reg] = b.sel(state.exec, bits, state.vreg.at(reg));
}
template <size_t N>
uint32_t descriptor_equal(SpirvCompute& b, RegState& state, const Rdna2Inst& in, int base,
                          const std::array<uint32_t, N>& descriptor) {
    auto equal = b.btrue();
    for (uint32_t c = 0; c < N; ++c) {
        Operand operand; operand.kind = OperandKind::SGPR; operand.value = base + static_cast<int>(c);
        bool available = true;
        const auto bits = operand_bits(b, state, in, operand, &available);
        if (!available) return 0; // MUST/lifetime filtering, not Function-variable presence
        equal = and_(b, equal, b.ucmp(Op_IEqual, bits, b.uconst(descriptor[c])));
    }
    return equal;
}
} // namespace

void PacketResourceServices::begin(SpirvCompute& b) {
    uint32_t pointer_type = 0;
    failure_var = b.function_var(b.t_u32, pointer_type);
    failure_pc_var = b.function_var(b.t_u32, pointer_type);
    m0_var = b.function_var(b.t_u32, pointer_type);
    b.store_function(failure_var, b.uconst(0));
    b.store_function(failure_pc_var, b.uconst(UINT32_MAX));
    b.store_function(m0_var, b.uconst(input.parameter_cache.entry_m0_available
        ? input.parameter_cache.entry_m0 : 0)); // allocation only; MUST proof protects every read
    for (uint32_t image = 0; image < input.images.size(); ++image)
        b.declare_texture(16 + image, Dim_2D, false, false, false);
}

void PacketResourceServices::fail(SpirvCompute& b, uint32_t condition, uint32_t pc,
                                  FragmentPacketRuntimeFailure reason) {
    const auto old = b.load_function(b.t_u32, failure_var);
    const auto first = and_(b, condition, b.ucmp(Op_IEqual, old, b.uconst(0)));
    b.store_function(failure_var, b.sel(first, b.uconst(static_cast<uint32_t>(reason)), old));
    b.store_function(failure_pc_var, b.sel(first, b.uconst(pc), b.load_function(b.t_u32, failure_pc_var)));
}

int PacketResourceServices::emit(SpirvCompute& b, RegState& state, const Rdna2Inst& in) {
    const auto consumed = [&](uint32_t bad) { return and_(b, state.exec, bad); };
    const auto numeric_failure = [&](const PacketF32Result& result) {
        fail(b, consumed(result.nonfinite), in.pc, FragmentPacketRuntimeFailure::NonFinite);
        fail(b, consumed(result.overflow), in.pc, FragmentPacketRuntimeFailure::FiniteOverflow);
    };
    if (in.fmt == Rdna2Format::SOP1 && in.opcode == kSop1OpcodeMovB32 && in.dst.value == 124) {
        bool valid = true;
        const auto bits = operand_bits(b, state, in, in.src[0], &valid);
        if (!valid) return -1;
        b.store_function(m0_var, bits);
        state.sreg[124] = bits;
        return 1;
    }
    if (in.fmt == Rdna2Format::SOPP && in.opcode == 0x0c) return 1;
    if (in.fmt == Rdna2Format::SMEM) {
        const auto found = std::find_if(input.buffers.begin(), input.buffers.end(),
            [&](const auto& resource) { return resource.pc == in.pc; });
        if (found == input.buffers.end()) return -1;
        const auto equal = descriptor_equal(b, state, in, in.src[0].value, found->descriptor);
        if (!equal) return -1;
        // Scalar loads execute even when EXEC=0. Descriptor failure is a packet failure, not an
        // unobservable inactive-vector failure; read ONLY the separately owned bounded snapshot.
        fail(b, b.logical_not(equal), in.pc, FragmentPacketRuntimeFailure::DescriptorMismatch);
        for (uint32_t c = 0; c < (1u << (in.opcode - 8)); ++c)
            state.sreg[in.dst.value + c] = input_word(b, b.uconst(buffer_offsets.at(in.pc) + in.literal / 4 + c));
        return 1;
    }
    if (in.fmt == Rdna2Format::VOP2 && (in.opcode == 3 || in.opcode == 8)) {
        bool valid = true;
        const auto a = operand_bits(b, state, in, in.src[0], &valid);
        const auto c = operand_bits(b, state, in, in.src[1], &valid);
        if (!valid) return -1;
        const auto value = in.opcode == 3 ? packet_f32_add(b, a, c, input.invocation.float_mode)
                                         : packet_f32_mul(b, a, c, input.invocation.float_mode);
        numeric_failure(value);
        write_vector(b, state, in.dst.value, value.bits);
        return 1;
    }
    if (in.fmt == Rdna2Format::VOP1 && packet_special_f32_opcode(in.opcode)) {
        bool available = true;
        const auto source = operand_bits(b, state, in, in.src[0], &available);
        if (!available) return -1;
        const auto value = packet_f32_special(b, source, in.opcode, input.invocation.float_mode);
        fail(b, consumed(value.nonfinite), in.pc,
             FragmentPacketRuntimeFailure::SpecialNanOrNegativeRoot);
        write_vector(b, state, in.dst.value, value.bits);
        return 1;
    }
    if (in.fmt == Rdna2Format::VINTRP) {
        const auto m0 = b.load_function(b.t_u32, m0_var);
        fail(b, consumed(b.ucmp(Op_INotEqual, m0, b.uconst(input.parameter_cache.m0))),
            in.pc, FragmentPacketRuntimeFailure::M0Mismatch);
        const auto coefficient = b.ibin(Op_IAdd, b.uconst(parameter_offsets.at(in.pc)),
            b.ibin(Op_IMul, b.linear_localid, b.uconst(3)));
        const auto p0 = input_word(b, coefficient);
        const auto delta = input_word(b, b.ibin(Op_IAdd, coefficient, b.uconst(in.opcode == 0 ? 1 : 2)));
        const auto barycentric = state.vreg.at(in.src[0].value);
        const auto old = state.vreg.at(in.dst.value);
        const auto product = packet_f32_mul(b, delta, barycentric, input.invocation.float_mode);
        const auto sum = packet_f32_add(b, product.bits, in.opcode == 0 ? p0 : old, input.invocation.float_mode);
        // Published P1/P2 equations alone do not settle fused/staged implementation. Admit only
        // the domain in which EACH product and addition is exactly representable and normal/zero.
        // Actual reaching VDST, not a nominal remembered P1, supplies P2's implicit source.
        auto exact = and_(b, normal_or_zero(b, barycentric), and_(b, product.valid, product.exact));
        exact = and_(b, exact, and_(b, normal_or_zero(b, product.bits), and_(b, sum.valid, sum.exact)));
        exact = and_(b, exact, normal_or_zero(b, sum.bits));
        // Exact nonzero sums are independent of staged/fused rounding. A final zero also needs
        // interpolation-specific signed-zero/rounding authority, which this first slice lacks.
        exact = and_(b, exact, b.ucmp(Op_INotEqual,
            b.ibin(Op_BitwiseAnd, sum.bits, b.uconst(0x7fffffffu)), b.uconst(0)));
        if (in.opcode == 1) exact = and_(b, exact, normal_or_zero(b, old));
        fail(b, consumed(b.logical_not(exact)), in.pc, FragmentPacketRuntimeFailure::InterpolationNotExact);
        write_vector(b, state, in.dst.value, sum.bits);
        return 1;
    }
    if (in.fmt == Rdna2Format::MIMG) {
        const auto found = std::find_if(input.images.begin(), input.images.end(),
            [&](const auto& resource) { return resource.pc == in.pc; });
        if (found == input.images.end()) return -1;
        const uint32_t binding = 16 + static_cast<uint32_t>(found - input.images.begin());
        const auto image_equal = descriptor_equal(b, state, in, in.src[1].value, found->descriptor);
        const auto sampler_equal = descriptor_equal(b, state, in, in.src[2].value, found->sampler);
        if (!image_equal || !sampler_equal) return -1;
        fail(b, consumed(b.logical_not(and_(b, image_equal, sampler_equal))), in.pc,
             FragmentPacketRuntimeFailure::DescriptorMismatch);
        const auto u = state.vreg.at(in.src[0].value), v = state.vreg.at(in.src[0].value + 1);
        const auto lod = in.opcode == 0x27 ? b.uconst(0) : state.vreg.at(in.src[0].value + 2);
        auto valid_lod = b.bfalse(), valid_coordinate = b.bfalse();
        // Nearest sampling at exact texel centers, with exact integral LODs, is the bounded initial
        // cross-architecture sampling envelope. No interpolation-precision or subnormal-entry claim.
        for (uint32_t level = 0; level < found->mips.size(); ++level) {
            const auto selected = b.ucmp(Op_IEqual, lod, b.uconst(fbits(static_cast<float>(level))));
            valid_lod = b.ucmp(Op_LogicalOr, valid_lod, selected);
            const auto& mip = found->mips[level];
            const auto valid_u = exact_center(b, u, mip.width), valid_v = exact_center(b, v, mip.height);
            valid_coordinate = b.ucmp(Op_LogicalOr, valid_coordinate, and_(b, selected, and_(b, valid_u, valid_v)));
        }
        fail(b, consumed(b.logical_not(valid_lod)), in.pc, FragmentPacketRuntimeFailure::SampleLodDomain);
        fail(b, consumed(b.logical_not(valid_coordinate)), in.pc, FragmentPacketRuntimeFailure::SampleCoordinateDomain);
        const auto safe = and_(b, state.exec, and_(b, valid_lod, valid_coordinate));
        uint32_t values[4];
        // No generic compute helper or ambient coordinate-normalization read: use EXACT explicit
        // sampler operation. Sanitized error/inactive operands are internal only; failures forbid
        // ALL export publication and inactive destinations retain their supplied old words.
        const auto sampler = b.id(), coordinate = b.id(), sampled = b.id();
        b.put(b.code, Op_Load, {b.tex_binding_simg.at(binding), sampler, b.tex_var.at(binding)});
        const auto safe_u = b.uconst(fbits(0.5f / static_cast<float>(found->mips.front().width)));
        const auto safe_v = b.uconst(fbits(0.5f / static_cast<float>(found->mips.front().height)));
        b.put(b.code, Op_CompositeConstruct, {b.t_v2f(), coordinate,
            b.bcf(b.sel(safe, u, safe_u)), b.bcf(b.sel(safe, v, safe_v))});
        b.put(b.code, Op_ImageSampleExplicitLod, {b.t_v4f, sampled, sampler, coordinate,
            ImgOp_Lod, b.bcf(b.sel(safe, lod, b.uconst(0)))});
        for (uint32_t c = 0; c < 4; ++c) {
            const auto value = b.id(); b.put(b.code, Op_CompositeExtract, {b.t_f32, value, sampled, c});
            values[c] = b.bcu(value);
            write_vector(b, state, in.dst.value + c, values[c]);
        }
        return 1;
    }
    return 0;
}

void PacketResourceServices::finish(SpirvCompute& b) {
    b.store_output_word(b.uconst(kFragmentResourceStatusMagic), kFragmentResourceStatusWords, output.status_offset, 0);
    b.store_output_word(b.load_function(b.t_u32, failure_pc_var), kFragmentResourceStatusWords, output.status_offset + 1, 0);
    b.store_output_word(b.load_function(b.t_u32, failure_var), kFragmentResourceStatusWords, output.status_offset + 2, 0);
}
} // namespace prosper::gpu
