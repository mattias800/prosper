# Offline GPU replay

Consumes frozen capture files for execution and read-only inspection, descriptor validation and
dependency graphs. Execution, raw regeneration, overrides and exports retain strict materialization.
Terminal metadata reports use an opaque observation whose normalized storage cannot be promoted
to a frame. Keep one formatter and one byte/descriptor normalizer for both paths.
