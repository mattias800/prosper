// Retain a defined wider cohort across a narrower same-lane accumulating write.
#pragma once
namespace prosper::gpu {
struct RegState;
struct Rdna2Inst;
struct ScalarSourceMarks;
// Run the ordinary transfer once; refine only its owner, keeping every root/flag plane.
void record_vector_word_marks_preserving_cohort(RegState& rs, const Rdna2Inst& in,
                                                const ScalarSourceMarks& source);
} // namespace prosper::gpu
