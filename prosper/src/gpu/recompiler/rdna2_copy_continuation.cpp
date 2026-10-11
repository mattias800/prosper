#include "gpu/recompiler/rdna2_copy_continuation.hpp"

#include "gpu/recompiler/rdna2_native_logical_words.hpp"
#include "gpu/recompiler/rdna2_saved_mask_words.hpp"
#include <algorithm>
#include <exception>
#include <limits>
#include <utility>

namespace prosper::gpu {
namespace {
template <class Cleanup>
class ScopedCleanup {
    Cleanup cleanup_;
    int exceptions_ = std::uncaught_exceptions();
    bool unwind_only_;

public:
    ScopedCleanup(Cleanup cleanup, bool unwind_only) noexcept
        : cleanup_(std::move(cleanup)), unwind_only_(unwind_only) {}
    ~ScopedCleanup() noexcept {
        if (!unwind_only_ || std::uncaught_exceptions() > exceptions_) cleanup_();
    }
    ScopedCleanup(const ScopedCleanup&) = delete;
    ScopedCleanup& operator=(const ScopedCleanup&) = delete;
};
} // namespace

class SavedMaskCopyFrame {
    friend class SavedMaskCopyBodyScope;
    friend bool consume_saved_mask_copy_continuation(SpirvCompute&, const std::vector<Rdna2Inst>&,
                                                     const uint32_t*, size_t, uint32_t,
                                                     const RegState*);

    struct Origin {
        std::vector<uint32_t> words;
        std::vector<Rdna2Inst> instructions;
        std::array<uint32_t, 10> domain;
        std::set<int> entry_masks;
        std::unordered_set<int> entry_b32_masks;
        std::set<uint32_t> native_copies, fragment_copies;
    };
    std::shared_ptr<const Origin> origin;
    SpirvCompute* builder = nullptr;
    RegState* state = nullptr;
    const std::vector<Rdna2Inst>* instructions = nullptr;
    const uint32_t* code = nullptr;
    size_t dwords = 0, begin = 0;
    uint64_t epoch = 0;
    bool open = true, authorized = false, initializer_permit = false;
    const std::vector<Rdna2Inst>* pending = nullptr;
    size_t pending_begin = 0;

    bool binding(const SpirvCompute& b, const std::vector<Rdna2Inst>& ins, const uint32_t* raw,
                 size_t width, const RegState* rs) const {
        if (!open || !authorized || builder != &b || state != rs || instructions != &ins ||
            code != raw || dwords != width || !origin || !raw ||
            epoch != b.saved_mask_copy_revocation_epoch || epoch == UINT64_MAX ||
            origin->domain != b.native_vcc_copy_entry_domain ||
            origin->entry_masks != b.native_vcc_copy_entry_masks ||
            origin->entry_b32_masks != b.native_vcc_copy_entry_b32_masks ||
            origin->native_copies != b.native_bool_only_vcc_copy_pcs ||
            origin->fragment_copies != b.fragment_bool_only_vcc_copy_pcs ||
            !native_vcc_copy_execution_domain_matches(b) || width != origin->words.size() ||
            b.saved_mask_original_words != origin->words ||
            !std::equal(origin->words.begin(), origin->words.end(), raw) ||
            begin > origin->instructions.size() ||
            ins.size() != origin->instructions.size() - begin)
            return false;
        return std::equal(ins.begin(), ins.end(), origin->instructions.begin() + begin,
                          [](const auto& in, const auto& original) {
                              return !in.synthetic_terminator && in == original;
                          });
    }

    bool closed_suffix(size_t first) const {
        if (!origin || first >= origin->instructions.size() || !origin->instructions.back().is_end)
            return false;
        const auto start = origin->instructions.begin() + first;
        for (auto it = start; it != origin->instructions.end(); ++it) {
            const auto& in = *it;
            if (in.synthetic_terminator || in.fmt == Rdna2Format::Unknown ||
                rdna2_escapes_decoded_effects(in))
                return false;
            if (in.fmt == Rdna2Format::SOPP && sopp_opcode_is_direct_branch(in.opcode)) {
                const int64_t target = int64_t(in.pc) + in.len_dwords + in.simm16;
                if (target < start->pc ||
                    std::none_of(
                        start, origin->instructions.end(),
                        [target](const auto& next) { return target == next.pc; }))
                    return false;
            }
        }
        return true;
    }
};

void revoke_saved_mask_copy_proof(SpirvCompute& b) {
    b.native_bool_only_vcc_copy_pcs.clear();
    b.fragment_bool_only_vcc_copy_pcs.clear();
    b.fragment_bool_only_saved_mask_pcs.clear();
    if (b.saved_mask_copy_revocation_epoch != std::numeric_limits<uint64_t>::max())
        ++b.saved_mask_copy_revocation_epoch;
}

bool consume_saved_mask_copy_continuation(SpirvCompute& b, const std::vector<Rdna2Inst>& ins,
                                          const uint32_t* code, size_t dwords,
                                          uint32_t initial_active, const RegState* initial) {
    const auto frame = b.saved_mask_copy_frame;
    if (!frame || !std::exchange(frame->initializer_permit, false) || initial_active) return false;
    return frame->binding(b, ins, code, dwords, initial) && frame->closed_suffix(frame->begin);
}

SavedMaskCopyBodyScope::SavedMaskCopyBodyScope(SpirvCompute& b, RegState& rs,
                                               const std::vector<Rdna2Inst>& ins,
                                               const uint32_t* code, size_t dwords,
                                               uint32_t initial_active)
    : builder_(b), state_(rs), instructions_(ins), code_(code), dwords_(dwords),
      previous_(std::exchange(b.saved_mask_copy_frame, {})) {
    ScopedCleanup rollback(
        [this, &b]() noexcept {
            if (frame_) frame_->open = false;
            b.saved_mask_copy_frame = previous_;
            revoke_saved_mask_copy_proof(b);
        },
        true);
    // Consume the parent's offer before initialization, callbacks or fallback routing. The
    // vector lives in the private emitter method through this entire child call.
    if (previous_ && previous_->builder == &b && previous_->state == &rs) {
        const auto* pending = std::exchange(previous_->pending, nullptr);
        if (pending == &ins && previous_->binding(b, *previous_->instructions, code, dwords, &rs) &&
            previous_->pending_begin > previous_->begin &&
            previous_->closed_suffix(previous_->pending_begin)) {
            frame_ = std::shared_ptr<SavedMaskCopyFrame>(new SavedMaskCopyFrame);
            frame_->origin = previous_->origin;
            frame_->begin = previous_->pending_begin;
            frame_->epoch = previous_->epoch;
            frame_->builder = &b;
            frame_->state = &rs;
            frame_->instructions = &ins;
            frame_->code = code;
            frame_->dwords = dwords;
            frame_->authorized = true;
            frame_->initializer_permit = true;
            b.saved_mask_copy_frame = frame_;
        }
    }
    initialize_saved_mask_word_proof(b, ins, code, dwords, initial_active, &rs);
    if (frame_) {
        frame_->authorized =
            !frame_->initializer_permit && frame_->binding(b, ins, code, dwords, &rs);
        frame_->initializer_permit = false;
        return;
    }
    // Established public first-full-program analysis is unchanged. This snapshot owns the
    // immutable bytes, decoded fields and entry proof independently of later caller mutations.
    if (!code || dwords != b.saved_mask_original_words.size() ||
        !std::equal(b.saved_mask_original_words.begin(), b.saved_mask_original_words.end(), code) ||
        !native_vcc_copy_entry_masks_match(b, &rs) ||
        (b.native_bool_only_vcc_copy_pcs.empty() && b.fragment_bool_only_vcc_copy_pcs.empty()) ||
        b.saved_mask_copy_revocation_epoch == UINT64_MAX)
        return;
    auto origin = std::make_shared<SavedMaskCopyFrame::Origin>();
    origin->words = b.saved_mask_original_words;
    origin->domain = b.native_vcc_copy_entry_domain;
    origin->entry_masks = b.native_vcc_copy_entry_masks;
    origin->entry_b32_masks = b.native_vcc_copy_entry_b32_masks;
    origin->native_copies = b.native_bool_only_vcc_copy_pcs;
    origin->fragment_copies = b.fragment_bool_only_vcc_copy_pcs;
    rdna2_walk(origin->words.data(), origin->words.size(), origin->instructions);
    if (ins.size() != origin->instructions.size() ||
        !std::equal(ins.begin(), ins.end(), origin->instructions.begin(),
                    [](const auto& in, const auto& original) {
                        return !in.synthetic_terminator && in == original;
                    }))
        return;
    frame_ = std::shared_ptr<SavedMaskCopyFrame>(new SavedMaskCopyFrame);
    frame_->origin = std::move(origin);
    frame_->builder = &b;
    frame_->state = &rs;
    frame_->instructions = &ins;
    frame_->code = code;
    frame_->dwords = dwords;
    frame_->epoch = b.saved_mask_copy_revocation_epoch;
    frame_->authorized = true;
    b.saved_mask_copy_frame = frame_;
}

SavedMaskCopyBodyScope::~SavedMaskCopyBodyScope() {
    if (frame_) frame_->open = false;
    builder_.saved_mask_copy_frame = previous_;
}

bool SavedMaskCopyBodyScope::emit_counted_postlude(
    uint32_t exit_pc, const std::unordered_set<uint32_t>& safe, const ShaderResourceTable* rt,
    bool allow_exec_update, bool allow_smem,
    const std::function<bool(RegState&, const Rdna2Inst&)>& exp_fn) {
    ScopedCleanup clear_offer(
        [this]() noexcept {
            if (frame_) frame_->pending = nullptr;
        },
        false);
    std::vector<Rdna2Inst> postloop;
    for (const auto& in : instructions_)
        if (in.pc >= exit_pc) postloop.push_back(in);
    if (postloop.empty()) return true;
    // CONFIDENCE: HIGH -- this seam is called only after the real counted loop's PHIs, exit
    // slots and marks succeeded. It preserves the original Bool-copy generation proof alone.
    // Physical producer participation still expires in initialize_saved_mask_word_proof.
    if (frame_ && frame_->binding(builder_, instructions_, code_, dwords_, &state_)) {
        const auto& original = frame_->origin->instructions;
        const auto first = std::find_if(original.begin() + frame_->begin, original.end(),
                                        [exit_pc](const auto& in) { return in.pc == exit_pc; });
        if (first != original.end()) {
            const size_t begin = static_cast<size_t>(first - original.begin());
            if (begin > frame_->begin && frame_->closed_suffix(begin)) {
                frame_->pending = &postloop;
                frame_->pending_begin = begin;
            }
        }
    }
    return emit_body(builder_, state_, postloop, safe, rt, allow_exec_update, allow_smem, exp_fn,
                     code_, dwords_);
}
}   // namespace prosper::gpu
