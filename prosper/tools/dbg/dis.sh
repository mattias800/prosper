#!/bin/bash
# Disassemble a range of the DOLL eboot .text from /tmp/text.bin
# Usage: dis.sh <start_hex_off> <len_hex>  (offsets are eboot-relative)
START=$1
LEN=$2
objdump -D -b binary -m i386:x86-64 --start-address=$((START)) --stop-address=$((START + LEN)) /tmp/text.bin | tail -n +8
