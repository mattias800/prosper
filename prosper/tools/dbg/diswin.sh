#!/bin/bash
# diswin.sh <va_hex> <len_hex> — disassemble an eboot text window (DOLL PPSA17942).
# Text LOAD file offset base = 0x34050 for VA 0 (self_dump segment table).
VA=$(( $1 ))
LEN=$(( $2 ))
OFF=$(( VA + 0x34050 ))
dd if=/root/PPSA17942-app0/eboot.bin of=/tmp/diswin.bin bs=1 skip=$OFF count=$LEN 2>/dev/null
objdump -D -b binary -m i386:x86-64 --adjust-vma=$VA /tmp/diswin.bin | tail -n +8
