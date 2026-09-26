#!/usr/bin/env python3
import pathlib
p = pathlib.Path('.lane/extract_blocks.py')
s = p.read_text()
new_jobs = '''    {   # PROSPER_DS_SLICE_CENSUS
        'first': '        if (PROSPER_ENV_ON("PROSPER_DS_SLICE_CENSUS")) {',
        'dedent': 8,
        'signature': read_frag('dsslice_hdr.txt'),
        'replacement': [
            '        if (PROSPER_ENV_ON("PROSPER_DS_SLICE_CENSUS"))',
            '            report_ds_slice_census(ds_key, use_depth, depth_used_meaningfully);',
        ],
    },
    {   # PROSPER_DRAW_STATS
        'first': '    if (ds_active) {',
        'dedent': 4,
        'signature': read_frag('drawstats_hdr.txt'),
        'replacement': [
            '    if (ds_active)',
            '        report_draw_stats_funnel(dev, ds_stats_pool, ds_occ_pool, dv, draws,',
            '                                 batch_completed);',
        ],
    },
'''
marker = "    {   # the geometry-probe readback"
assert marker in s
assert "PROSPER_DS_SLICE_CENSUS" not in s
s = s.replace(marker, new_jobs + marker)
p.write_text(s)
print('ok')
