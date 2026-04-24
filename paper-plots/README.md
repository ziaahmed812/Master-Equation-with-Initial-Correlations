# Paper Parameter Scripts

This folder contains one script for each parameter set used in the published
figures. The scripts are deliberately plain: they define `SystemParams`,
`BathParams`, `tlist`, observables, numerical settings, and then call the same
public solvers used in the examples.

They are useful when you want to see exactly which package parameters
correspond to a published calculation.

What they do:

- compute solver arrays in memory
- keep the physical parameters visible in the script
- use the public `meic.solve(...)` and `meic.exact.solve(...)` APIs

What they do not do:

- save EPS or PNG files automatically
- provide a separate plotting API
- hide the calculation behind a figure id

Use them as transparent parameter recipes. After a script returns a `Result`,
plot or save the arrays however you prefer.
