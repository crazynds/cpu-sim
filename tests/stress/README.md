# Stress suite

58 CUDA programs, each a group of cases that check themselves against a host reference. Together they exercise the parts of CUDA real programs use: launch geometry, types, math, control flow, per-thread arrays, memory and the runtime API, atomics and synchronization, warp operations, cooperative groups, libcu++ atomics, graphs, and about 80 classic workloads (reductions, scans, sorts, stencils, linear algebra, graphs, physics, ML layers, dynamic programming).

Every program passes on a real GPU built with `nvcc`; that is what makes a failure on the simulator the simulator's.

## Running it

```sh
build/cpusimd --daemon                       # the suite runs on the driver
tests/stress/run.sh cpu                      # every program, built with cucc
tests/stress/run.sh gpu                      # every program, built with nvcc (the oracle)
tests/stress/run.sh both wl_scan sync_warp   # some programs, both ways
```

Results go to `build/stress/`: each program's output (`<name>.cpu.log`, `<name>.gpu.log`), the compiler's (`<name>.cpu.build.log`), and `summary.tsv`, one line per case:

| Column | Meaning |
|---|---|
| `status` | `PASS`; `FAIL` (wrong result); `COMPILE` (cucc rejected the program); `ERROR` (a CUDA call failed); `CRASH`; `TIMEOUT`; `DAEMON` (the driver died); `MISSING` (the program stopped before this case) |
| `kernel_ms` | the device time of the case, best of 3 after a warm-up |
| `ref_ms` | the same computation on the host with OpenMP, every thread (`-1`: no meaningful equivalent) |
| `gpu_ms` | `kernel_ms` of the GPU run, when there is one |

`kernel_ms / ref_ms` is how the suite finds slow cases: above 1, the CUDA version on the simulator is slower than plain C++ with OpenMP on the same CPU.

Options, by environment variable: `STRESS_TIMEOUT` (seconds per program, default 120), `STRESS_OUT` (another output directory), `STRESS_BUILD` (another build to test, with `CPUSIM_SOCKET` for its driver).

## Writing a program

Include `stress.cuh`, check each case against a host computation, report it with `stress::report(name, ok, kernel_ms, ref_ms)` and end with `return stress::finish();`. Keep one feature area per file, so that something the simulator cannot compile hides only its own cases, and check it on a GPU first: `tests/stress/run.sh gpu <name>`.
