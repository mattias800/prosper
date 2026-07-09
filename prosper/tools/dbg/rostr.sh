#!/bin/bash
# rostr.sh <va_hex_eboot_relative>... — print eboot rodata strings (DOLL PPSA17942)
for va in "$@"; do
  off=$(( 0x66e04d0 + 0x$va - 0x669c000 ))
  echo -n "va 0x$va: "
  dd if=/root/PPSA17942-app0/eboot.bin bs=1 skip=$off count=160 status=none | tr "\0" "." | head -c 160
  echo
done
