# SELF and ELF modules

This folder parses plaintext SELF containers and raw ELF modules into host-independent
segments, dynamic tables, symbols and relocations, then builds their relocatable memory
images. SELF data segments resolve container file offsets; program headers define the
loaded image. Keep parser and image errors visible to the loader's callers.

Multi-module linking and import resolution belong in `src/loader`; host mappings and
execution belong in `src/host/image`. Synthetic fixtures can exercise this folder without
executing guest code or reading game dumps.
