# Tracking runs (Xeon Gold 6548Y+ server, daemon, least of 3 runs)

| Run | Build | Notes |
|---|---|---|
| `base3` | 7436e9a reverted to the previous code generation, kernels compiled apart | baseline |
| `classes3` | cdb709e | 1D shape classes |
| `nocoop2` | this commit | cooperative variants removed; Job's hot atomics on lines of their own |
| `w1.features`, `w1.widths` | cdb709e | kernel features and measured times per width (widths.py) |

Compare two with `../compare.py <before> <after>`. Programs that vary run to run by 20% or more: reverse, heat2d, quantAQLM, is.
