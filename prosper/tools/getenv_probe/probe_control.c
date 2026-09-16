// Positive control for getenv_probe. Verifies the probe on a KNOWN answer before any run of it
// is believed: 200,000 iterations x 3 names per second for 12 s, so the 10-second report must
// read exactly total=6000000 with three names at 2,000,000 each.
//
// It exists because the obvious hand-check does not work. Running the probe under `bash -c` and
// expanding $HOME in a loop produces total=0 -- bash reads its own variable table and never calls
// getenv -- and a by-name census that writes an empty file is indistinguishable from one whose
// table is broken. Build and run:
//
//     gcc -O2 -o probe_control probe_control.c
//     LD_PRELOAD=$PWD/probe.so PROSPER_GETENV_PROBE_NAMES=$PWD ./probe_control
//     sort -rn getenv-names-*.txt | head
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
int main(void) {
    for (int t = 0; t < 12; ++t) {
        for (int i = 0; i < 200000; ++i) {
            (void)getenv("PROSPER_CONTROL_A");
            (void)getenv("PROSPER_CONTROL_B");
            (void)getenv("HOME");
        }
        sleep(1);
    }
    printf("control done\n");
    return 0;
}
