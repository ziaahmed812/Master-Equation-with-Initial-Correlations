# Paper Parameter Scripts

This folder contains one script for each published paper parameter set. The
scripts intentionally use the same black-box API as ordinary users:

- define `SystemParams`
- define `BathParams`
- define `tlist`
- choose `e_ops`
- call `meic.solve(...)` or `meic.exact.solve(...)` with explicit correlation choice
- leave plotting and saving to the user

Despite the historical folder name, these scripts do not generate EPS or PNG
files. They compute arrays in memory so users can inspect, save, or plot them
however they prefer.
