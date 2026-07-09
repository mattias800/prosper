#!/bin/bash
# Disassemble the AGC import thunks to reveal their GOT slots.
for a in 0x58deee0 0x58def30 0x58def70 0x58defc0 0x58df020 0x58df440 0x58df4b0 0x58df510 \
         0x58dfa10 0x58dfa90 0x58dfd30 0x58dfc30 0x58dfaf0 0x58dfae0 0x58dfb00 0x58dfc00 0x58fdb00; do
  echo "== $a =="
  objdump -D -b binary -m i386:x86-64 --start-address=$((a)) --stop-address=$((a + 0x14)) /tmp/text.bin | tail -n +8 | head -4
done
