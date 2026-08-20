# `present` — scanout

`videoout_present` — taking a finished frame to the display path.

The last stage, and therefore the one where an upstream failure is *observed* rather than caused. A
black or stale frame here is almost never a bug here. Check that something was drawn, and that the
target being presented is the target that was drawn to, before looking for a defect in this folder.
