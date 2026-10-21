# Iteration 4

The fourth round, on the two-socket Xeon with the H100 next to it: compile times of minutes brought to seconds without slowing any kernel down, a variant that never won removed (and the false sharing that removal exposed), an intermittent failure traced to the kernel's address randomization, the width selection rebuilt to decide the same way every time, the code split by subject with tests for each of these, and a first comparison on programs drawn at random instead of chosen.

| # | Document | In one line |
|---|---|---|
| 35 | [Compile time](35-compile-time.md) | Kernels compiled in processes of their own, classes of block shapes, loops with a barrier kept rolled, no 128-lane variants past 64 live values: sc 499 to 3.3 s, hadamard 1200 to 29 s |
| 36 | [A variant that never won, a false sharing, a failure one run in fifty](36-cooperative-variant-false-sharing-and-vram.md) | The SMT cooperative variant removed; the job's hot atomics on lines of their own; the VRAM below where PIE executables load |
| 37 | [The width selection, the same way every time](37-width-selection.md) | Critical path in CPU time, a warm-up, up to 15 rounds estimated from their lower half, ties within 25% by order, early stop, short launches unmeasured, the measurement kept across programs: 65 of 73 kernels within 10% of the best, fresh drivers alike on all but 3 to 6 of 51 |
| 38 | [Where we stand](38-where-we-stand.md) | 72 HeCBench programs drawn at random: 52 of 52 pass; faster than OpenMP in 15 of 27; 7.8x behind an H100 |
| 39 | [The code by subject, and tests](39-organization-and-tests.md) | The SIMT pass, the runtime and the compiler driver split by subject; tests for every problem this round fixed |

## Open points

- **Transcendentals in single precision** are computed in double for CUDA's ulp bounds; bilateral, with 49 `expf` per pixel, runs 73x behind the H100 ([38](38-where-we-stand.md)).
- **Atomics on one address from every thread:** increment and decrement with wrap, exchange and compare-and-swap are not merged per warp, and merged ones still contend across all cores ([38](38-where-we-stand.md)).
- **Grids made for the GPU** with most threads idle and a stream of memory per thread (memtest 175x behind OpenMP) ([38](38-where-we-stand.md)).
- **Predicting the widths worth building** from a kernel's features, to compile fewer variants ([35](35-compile-time.md)).
- **The width chosen by fresh drivers** still differs for 3 to 6 of 51 kernel shapes, whose best widths are close to the tie margin ([37](37-width-selection.md), [known issues](../../known-issues.md)).
- From earlier rounds: separate compilation of device code (`-dc`), inline PTX, and ECL-BH's octree build (HeCBench bh) ([known issues](../../known-issues.md)).

Previous: [iteration 3](../iteration-3/README.md). Back to the [index](../README.md).
