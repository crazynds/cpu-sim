# 6. Tracking performance, the width selection and stability

Tutorials 2 to 5 measure single programs in depth. This one measures the simulator across many: whether a change to the compiler or the executor made programs faster or slower, how close the automatic width selection comes to the best width, whether fresh drivers decide alike, and whether a failure seen once comes back. Every script here starts a driver of its own on a private socket, so it can run next to another driver, and leaves nothing running.

They use HeCBench ([github.com/zjin-lcf/HeCBench](https://github.com/zjin-lcf/HeCBench)), unmodified: set `HECBENCH` to its `src` directory (default `~/fire/src`). Each program is compiled from its `-cuda` directory with its own Makefile (`make CC=<build>/cucc ARCH=sm_70`) and run with the arguments in `bench/track/apps.txt` or, when none are listed there, its Makefile's `run` target.

## Performance across builds: `track.sh` and `compare.py`

```sh
bench/track/track.sh build before          # results/before.csv
# ... change the compiler or the executor, rebuild ...
bench/track/track.sh build after
bench/track/compare.py before after
```

`track.sh <build> <label> [apps.txt]` compiles the 49 programs of `bench/track/apps.txt` and runs each `REPS` times (3 by default) with `CPUSIM_KERNEL_TIMES=1`, which makes the runtime print every kernel's launches and time at exit (`cpusim-times <kernel> <launches> <ns>` and `cpusim-times-total`). One CSV line per program: compile seconds, binary size, `ok` or `FAIL` (exit status or a `FAIL`/`mismatch` line), the least of the runs' total kernel time, and the wall time of all runs. A failing run's output is kept as `results/<label>.<app>.fail<n>.log`.

`compare.py <before> <after>` prints, per program, the kernel time ratio (marked `slower`/`faster` beyond 10%), compile time and size, and the geometric means. Read the ratios with the noise in mind: some programs (reverse, heat2d, haccmk) vary by 20 to 40% between runs of the same build, and a whole machine shared with other work can slow down for minutes. A change is real when its programs move together and the binary sizes say the code changed; `compare.py` lists sizes too, which tells a code change from noise.

## The width selection against every width: `oracle.sh` and `oracle.py`

```sh
MODES="auto 32 32r 64 64s 128 128s 256 512" REPS=3 bench/track/oracle.sh build o1
bench/track/oracle.py o1
```

`oracle.sh <build> <label> [apps.txt]` compiles every program once, then runs each program in every mode, back to back, on a fresh driver per mode (the machine's load changes over minutes, and the modes of one program must see the same): `auto` is the automatic selection, any other mode is a width forced with `CPUSIM_LANES` (read by the driver). A forced width the kernel has no variant for falls back to 32 lanes, so such a line repeats the 32-lane one. One line per mode, program and kernel in `results/o1.oracle.tsv`, the least of `REPS` runs; the driver's width messages per mode in `results/o1.oracle.<mode>.widths`.

`oracle.py <label>` compares, per kernel, the automatic selection with the best forced width, and counts kernels by how far off it is (within 5%, 5 to 10%, ... over 2x), leaving out kernels under 1 ms (`--min-ms`). A mode may repeat as `auto#2`: the automatic selection again on another fresh driver.

Two cautions. The forced widths are measured minutes apart, so a kernel that varies between runs shows up as a miss: when two forced widths that run the same code (one of them a fallback) differ, the difference is noise. And since the selection's decisions live in the driver, `auto` in the same driver as a previous run of the program measures nothing: each mode gets a fresh driver for that reason.

## Do fresh drivers decide alike: `stability.py`

```sh
MODES="auto auto#2 auto#3 auto#4 auto#5" REPS=1 bench/track/oracle.sh build s1
bench/track/stability.py s1
```

Five fresh drivers, one run of each program on each. Each mode's width messages go to `results/s1.oracle.<mode>.widths`. `stability.py` reads the five width logs and lists every kernel and block shape that got more than one width, then the count. A kernel listed with widths whose estimates (in the logs) are close to the 25% tie margin is a near tie; one whose gap between the same two widths changes from driver to driver points at interference from other work on the machine.

## A failure seen once: `repeat.sh`

```sh
bench/track/repeat.sh build 30 groupnorm concurrentKernels
```

Compiles each program and runs it 30 times on one driver, printing how many runs failed; each failing run's output is kept in `results/repeat/<app>.<n>.log`. A failure that does not come back in 30 runs goes to [docs/known-issues.md](../known-issues.md) with what was seen and where.

## Against the GPU and OpenMP: `bench/overview/run.sh`

```sh
bench/overview/run.sh build <src dir> out/ app1 app2 ...
```

For programs that have both a `-cuda` and an `-omp` directory in `<src dir>`: builds each with `nvcc` (H100, `sm_90`), `cucc` and clang with OpenMP (`OMPCC`, a wrapper that drops the offload flags), timing each build, and runs the three `RUNS` times (2) with a limit of `TMO` seconds (120). `out/results.csv` has one line per program and form: build status and seconds, binary size, runs that succeeded, best wall time and the program's own verdict; each program's output is in `out/<app>.<form>.txt`, where its own kernel time lines are. [38](../optimization/iteration-4/38-where-we-stand.md) is this script on 72 programs drawn at random; their lists and results are in `bench/overview/results/`.

## Kernel features and widths: `collect_widths.sh` and `widths.py`

`collect_widths.sh <build> <label>` gathers, for every tracked program, each kernel's features as the compiler sees them (`CPUSIM_DEVC_FEATURES=1`: live 32-bit units, instructions, loads, stores, slow operations, loops, barriers) and the times the runtime measured for each width. `widths.py` joins them. This is the data behind the limit of 64 live units for variants of 128 lanes and more ([35](../optimization/iteration-4/35-compile-time.md)) and the removal of the cooperative variant ([36](../optimization/iteration-4/36-cooperative-variant-false-sharing-and-vram.md)).
