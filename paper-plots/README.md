# Paper Parameter Scripts

This folder contains one script for each published paper parameter set. The
scripts intentionally use the same black-box API as ordinary users:

- define `SystemParams`
- define `BathParams`
- define `tlist`
- choose `e_ops`
- pin the paper coefficient, bath-correlation, and initial-state cutoff settings explicitly with `NumericsConfig`
- call `meic.solve(...)` or `meic.exact.solve(...)` with explicit correlation choice
- leave plotting and saving to the user

These scripts do not generate EPS or PNG files. They compute arrays in memory
so users can inspect, save, or plot them however they prefer. They are written
as executable notebook-style parameter scripts, not import-safe library modules
or full-resolution internal figure-regeneration jobs.
