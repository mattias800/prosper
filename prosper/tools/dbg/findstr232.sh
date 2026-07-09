#!/bin/bash
# findstr232.sh <pattern> — locate UTF-16LE strings in DOLL eboot rodata, print VA.
# VA = 0x669c000 + (file_off - 0x66e04d0)
strings -e l -t x /root/PPSA17942-app0/eboot.bin | grep -E "$1" | head -${2:-20} | while read off s; do
  va=$(( 0x669c000 + 0x$off - 0x66e04d0 ))
  printf "va=0x%x  %s\n" $va "$s"
done
