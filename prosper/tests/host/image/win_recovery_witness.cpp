// Independent physical-register witness: values are observed after restore, not read from its buffer.
#define SAVE_XMM(n, off) "movdqu %xmm" #n ", " #off "(%rsp)\n"
#define RESTORE_XMM(n, off) "movdqu " #off "(%rsp), %xmm" #n "\n"
#define CANARY_XMM(n)                                                                              \
    "movl $" #n ", %eax\n"                                                                         \
    "movd %eax, %xmm" #n "\n"                                                                      \
    "pshufd $0, %xmm" #n ", %xmm" #n "\n"
#define CLEAR_XMM(n) "pxor %xmm" #n ", %xmm" #n "\n"
#define OBSERVE_XMM(n, off) "movdqu %xmm" #n ", " #off "(%r10)\n"

// Normal MS-x64 call, with an independent save of this fixture's own caller state. No instruction
// here reads the production recovery layout. RBP is deliberately a canary, not a frame address.
// clang-format off: adjacent instruction literals and register macros must stay reviewable as assembly.
__asm__(
    ".text\n.p2align 4\n.globl prosper_test_win_recovery_witness\n"
    "prosper_test_win_recovery_witness:\n"
    "pushq %rbp\n pushq %rbx\n pushq %rdi\n pushq %rsi\n"
    "pushq %r12\n pushq %r13\n pushq %r14\n pushq %r15\n"
    "subq $232, %rsp\n"
    "movq %rcx, 32(%rsp)\n movq %rdx, 40(%rsp)\n"
    "stmxcsr 48(%rsp)\n fnstcw 52(%rsp)\n"
    SAVE_XMM(6, 64) SAVE_XMM(7, 80) SAVE_XMM(8, 96) SAVE_XMM(9, 112)
    SAVE_XMM(10, 128) SAVE_XMM(11, 144) SAVE_XMM(12, 160) SAVE_XMM(13, 176)
    SAVE_XMM(14, 192) SAVE_XMM(15, 208)
    "movl $0x3f80, 56(%rsp)\n ldmxcsr 56(%rsp)\n"
    "movw $0x077f, 60(%rsp)\n fldcw 60(%rsp)\n"
    "movabsq $0x1111000011110001, %rbx\n"
    "movabsq $0x1111000011110002, %rbp\n"
    "movabsq $0x1111000011110003, %rdi\n"
    "movabsq $0x1111000011110004, %rsi\n"
    "movabsq $0x1111000011110005, %r12\n"
    "movabsq $0x1111000011110006, %r13\n"
    "movabsq $0x1111000011110007, %r14\n"
    "movabsq $0x1111000011110008, %r15\n"
    CANARY_XMM(6) CANARY_XMM(7) CANARY_XMM(8) CANARY_XMM(9) CANARY_XMM(10)
    CANARY_XMM(11) CANARY_XMM(12) CANARY_XMM(13) CANARY_XMM(14) CANARY_XMM(15)
    "movq 32(%rsp), %rcx\n callq prosper_win_recovery_save\n"
    "testl %eax, %eax\n jnz 1f\n"
    "xorl %ebx, %ebx\n xorl %ebp, %ebp\n xorl %edi, %edi\n xorl %esi, %esi\n"
    "xorl %r12d, %r12d\n xorl %r13d, %r13d\n xorl %r14d, %r14d\n xorl %r15d, %r15d\n"
    CLEAR_XMM(6) CLEAR_XMM(7) CLEAR_XMM(8) CLEAR_XMM(9) CLEAR_XMM(10)
    CLEAR_XMM(11) CLEAR_XMM(12) CLEAR_XMM(13) CLEAR_XMM(14) CLEAR_XMM(15)
    "movl $0x5f80, 56(%rsp)\n ldmxcsr 56(%rsp)\n"
    "movw $0x0b7f, 60(%rsp)\n fldcw 60(%rsp)\n"
    "movq 32(%rsp), %rcx\n callq prosper_win_recovery_restore\n ud2\n"
    "1:\n movq 40(%rsp), %r10\n"
    "movq %rbx, 0(%r10)\n movq %rbp, 8(%r10)\n"
    "movq %rdi, 16(%r10)\n movq %rsi, 24(%r10)\n"
    "movq %r12, 32(%r10)\n movq %r13, 40(%r10)\n"
    "movq %r14, 48(%r10)\n movq %r15, 56(%r10)\n"
    OBSERVE_XMM(6, 64) OBSERVE_XMM(7, 80) OBSERVE_XMM(8, 96) OBSERVE_XMM(9, 112)
    OBSERVE_XMM(10, 128) OBSERVE_XMM(11, 144) OBSERVE_XMM(12, 160) OBSERVE_XMM(13, 176)
    OBSERVE_XMM(14, 192) OBSERVE_XMM(15, 208)
    "stmxcsr 224(%r10)\n fnstcw 228(%r10)\n movq %rsp, 232(%r10)\n"
    "ldmxcsr 48(%rsp)\n fldcw 52(%rsp)\n"
    RESTORE_XMM(6, 64) RESTORE_XMM(7, 80) RESTORE_XMM(8, 96) RESTORE_XMM(9, 112)
    RESTORE_XMM(10, 128) RESTORE_XMM(11, 144) RESTORE_XMM(12, 160) RESTORE_XMM(13, 176)
    RESTORE_XMM(14, 192) RESTORE_XMM(15, 208)
    "addq $232, %rsp\n"
    "popq %r15\n popq %r14\n popq %r13\n popq %r12\n"
    "popq %rsi\n popq %rdi\n popq %rbx\n popq %rbp\n retq\n");
// clang-format on

#undef SAVE_XMM
#undef RESTORE_XMM
#undef CANARY_XMM
#undef CLEAR_XMM
#undef OBSERVE_XMM
