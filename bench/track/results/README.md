# Tracking runs (Xeon Gold 6548Y+ server, daemon, least of 3 runs)

| Run | Build | Notes |
|---|---|---|
| `base3` | 7436e9a reverted to the previous code generation, kernels compiled apart | baseline |
| `classes3` | cdb709e | 1D shape classes |
| `nocoop2` | this commit | cooperative variants removed; Job's hot atomics on lines of their own |
| `w1.features`, `w1.widths` | cdb709e | kernel features and measured times per width (widths.py) |
| `rolled` | d89d893 | loops with a barrier kept rolled (sc joins the tracked programs) |
| `heavy` | 7e19fae | no variants of 128 lanes or more past 64 live values (hadamard joins) |
| `o1.oracle.tsv` | f249765 | width oracle (oracle.sh), automatic selection of iteration 2 against every forced width |
| `o8.oracle.tsv` | b5cd2f8 | width oracle after the critical path, medians, tie rule and short launches |
| `s1`, `s4`, `s5`, `s6` `.oracle.auto*.widths` | eba7010 / b5cd2f8 / CPU time (loaded machine) / CPU time (quiet machine) | width choices of 5 fresh drivers each (stability.py) |
| `s7` to `s10` `.oracle.auto*.widths` | 15 rounds / ties within 25% / lower-half estimate / early stop at 20% (the selection now) | the same |
| `f1.oracle.tsv`, `f1.oracle.*.widths` | the selection now | width oracle with every mode of a program back to back, and 5 automatic modes |

Compare two with `../compare.py <before> <after>`. Programs that vary run to run by 20% or more: reverse, heat2d, quantAQLM, is.
