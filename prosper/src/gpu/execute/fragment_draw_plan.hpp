#pragma once
#include "gpu/recompiler/fragment_draw_gpu.hpp"
#include "gpu/execute/fragment_draw_residency.hpp"
#include <memory>
#include <span>

namespace prosper::gpu {
// An explicit emulator scheduling contract, not a reconstruction of AMD's undisclosed hardware
// append order. Complete genuine Vulkan quads remain consecutive and retain primitive identity.
// The first recipe additionally proves the original program insensitive to inter-quad placement
// and to private unexportable scratch workers in the last workgroup. Future recipes may support
// observable guest composition only with the additional authority they actually require.
enum class FragmentDrawScheduling : uint8_t { PackingUnobservableCompleteQuads };
enum class FragmentDrawEntryRecipe : uint8_t { OwnedUserPrefixAndShaderDefinedMasks };
struct FragmentDrawMaskWord {
    bool user_word = false;
    uint32_t word = 0; // owned prefix index, or original inline/literal word
};
struct FragmentDrawFullMask {
    uint32_t pc = 0;
    FragmentDrawMaskWord lo, hi;
};

class FragmentDrawTransaction;
class FragmentDrawProgramPlan {
public:
    FragmentDrawProgramPlan(const FragmentDrawProgramPlan&) = default;
    FragmentDrawProgramPlan(FragmentDrawProgramPlan&&) = default;
    FragmentDrawProgramPlan& operator=(const FragmentDrawProgramPlan&) = default;
    FragmentDrawProgramPlan& operator=(FragmentDrawProgramPlan&&) = default;
    const auto& capacity_owner() const { return capacity; }
    const auto& collector_shape() const { return collector; }
    const auto& collect_words() const { return collect; }
    const auto& count_words() const { return count; }
    const auto& assembly_words() const { return assemble; }
    const auto& validation_words() const { return validate; }
    const auto& replay_words() const { return replay; }
    const auto& replay_owner() const { return replay_shared; }
    const auto& rejection_reason() const { return rejection; }
    const auto& device_contract() const { return device; }
    bool source_live() const { return source_generations->live(); }

private:
    friend FragmentDrawProgramPlan compile_fragment_draw_program(const RasterQuadInputs&,
                                                                 const FragmentPacketPreparation&,
                                                                 FragmentPacketDeviceContract,
                                                                 uint32_t,
                                                                 RecompileDiagnosticContext);
    friend std::shared_ptr<const FragmentDrawProgramPlan>
    cached_fragment_draw_program(const RasterQuadInputs&, const FragmentPacketPreparation&,
                                 FragmentPacketDeviceContract, uint32_t,
                                 RecompileDiagnosticContext);
    friend FragmentDrawTransaction instantiate_fragment_draw_transaction(
        std::shared_ptr<const FragmentDrawProgramPlan>, std::shared_ptr<const RasterQuadInputs>,
        const FragmentPacketPreparation&, uint32_t, uint32_t, uint32_t, uint64_t);
    FragmentDrawProgramPlan() = default;
    // Only original-program compilation constructs this complete code/profile owner. Const
    // getters cannot turn a copied public plan into a different shader or attachment authority.
    std::shared_ptr<const FragmentDrawCapacity> capacity;
    // Weak source tracking cannot be kept alive by this plan's own copied ISA/SPIR-V payload.
    std::shared_ptr<FragmentDrawSourceGenerations> source_generations =
        std::make_shared<FragmentDrawSourceGenerations>();
    RasterQuadCollector collector;
    std::vector<uint32_t> collect, count, assemble, validate, replay;
    std::shared_ptr<const std::vector<uint32_t>> replay_shared;
    FragmentDrawScheduling scheduling = FragmentDrawScheduling::PackingUnobservableCompleteQuads;
    FragmentDrawEntryRecipe entry_recipe =
        FragmentDrawEntryRecipe::OwnedUserPrefixAndShaderDefinedMasks;
    // No per-draw scalar words, dimensions, primitive count or observed pixels belong here.
    // The original code, presence/profile/resource shape and actual enabled device own this plan.
    uint32_t user_prefix_presence = 0;
    uint32_t user_prefix_count = 0, rsrc2 = 0;
    FragmentFloatMode float_mode{};
    FragmentFloatFlags float_flags{};
    FragmentLaunchRsrc1 launch_rsrc1{};
    FloatTransportConfig transport{};
    FragmentPacketDeviceContract device{};
    std::vector<FragmentDrawFullMask> full_masks;
    std::string rejection;
};

class FragmentDrawTransaction {
public:
    FragmentDrawTransaction() = default;
    const std::shared_ptr<const FragmentDrawProgramPlan>& program() const { return program_; }
    const std::shared_ptr<const RasterQuadInputs>& producing_inputs() const {
        return producing_inputs_;
    }
    const std::vector<uint32_t>& entry_words() const { return entry_words_; }
    uint32_t width() const { return width_; }
    uint32_t height() const { return height_; }
    uint32_t primitive_count() const { return primitive_count_; }
    uint64_t executing_device_identity() const { return executing_device_identity_; }
    const std::string& rejection() const { return rejection_; }

private:
    friend FragmentDrawTransaction instantiate_fragment_draw_transaction(
        std::shared_ptr<const FragmentDrawProgramPlan>, std::shared_ptr<const RasterQuadInputs>,
        const FragmentPacketPreparation&, uint32_t, uint32_t, uint32_t, uint64_t);
    explicit FragmentDrawTransaction(std::string rejection) : rejection_(std::move(rejection)) {}
    FragmentDrawTransaction(std::shared_ptr<const FragmentDrawProgramPlan> program,
                            std::shared_ptr<const RasterQuadInputs> inputs,
                            std::vector<uint32_t> words, uint32_t width, uint32_t height,
                            uint32_t primitives, uint64_t device)
        : program_(std::move(program)), producing_inputs_(std::move(inputs)),
          entry_words_(std::move(words)), width_(width), height_(height),
          primitive_count_(primitives), executing_device_identity_(device) {}
    // The submitting API has no mutable metadata/upload copy. After instantiation neither
    // profile, full-mask conditions nor dynamic words can change before the private GPU upload.
    std::shared_ptr<const FragmentDrawProgramPlan> program_;
    std::shared_ptr<const RasterQuadInputs> producing_inputs_;
    std::vector<uint32_t> entry_words_;
    uint32_t width_ = 0, height_ = 0, primitive_count_ = 0;
    uint64_t executing_device_identity_ = 0;
    std::string rejection_;
};

// Program-only compilation, called from a source-generation calling-thread cache. Draw-dependent values
// must never be specialization keys or be substituted for unavailable guest entry values.
FragmentDrawProgramPlan
compile_fragment_draw_program(const RasterQuadInputs&, const FragmentPacketPreparation&,
                              FragmentPacketDeviceContract, uint32_t max_quads,
                              RecompileDiagnosticContext = {RecompileDiagnosticStage::Fragment, 0});
// Calling-thread code/profile cache: all live original generations remain resident; dead ones retire.
// Initial/user scalar VALUES never enter its key. There is no permanent shader-count admission cap.
std::shared_ptr<const FragmentDrawProgramPlan>
cached_fragment_draw_program(const RasterQuadInputs&, const FragmentPacketPreparation&,
                             FragmentPacketDeviceContract, uint32_t max_quads,
                             RecompileDiagnosticContext = {RecompileDiagnosticStage::Fragment, 0});
// No compile/parse/readback. Checks exact producing owners and dynamic launch/profile/presence,
// then retains a complete entry upload. Fixed-function attachment admission is a separate backend
// proof; this API cannot grant depth/stencil/clear, helper backing or resource epoch authority.
FragmentDrawTransaction instantiate_fragment_draw_transaction(
    std::shared_ptr<const FragmentDrawProgramPlan>, std::shared_ptr<const RasterQuadInputs>,
    const FragmentPacketPreparation&, uint32_t width, uint32_t height, uint32_t primitive_count,
    uint64_t executing_device_identity);
}   // namespace prosper::gpu
