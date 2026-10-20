# 39. The code by subject, and tests that pin this round down

> **In one sentence:** the three largest files (the SIMT pass, the runtime, the compiler's driver) are split by subject without changing a byte of what is generated, the width selection is a unit of its own that is tested without running a kernel, and every problem this round fixed has a test that fails if it comes back.

## Split by subject

| Before | After |
|---|---|
| `src/compiler/simt.cpp`, 4080 lines | `simt.cpp` (preparation, if-conversion, values, `run`), `simt_memory.cpp` (loads, stores, gathers, scatters), `simt_ops.cpp` (calls, atomics, warp operations, math, tensor cores), `simt_emit.cpp` (linearized control flow, barriers, profiling counters), around `simt_internal.h` |
| `src/compiler/main.cpp`, 1465 lines | `main.cpp` (836: options, the order of the passes, the variants, code generation) and `passes.{h,cpp}` (the IR passes around the SIMT pass) |
| `src/cudart/runtime.cpp`, 2187 lines | `runtime.cpp` (device, streams, events, errors, settings), `module.cpp` (registration, manifests, JIT, symbols), `launch.cpp` (launches, attributes, occupancy, profiling), `memory.cpp` (allocations and copies), `graph.cpp` (graphs and capture), around `runtime_internal.h` |
| width selection inside `DeviceCore` | `src/device/width_tuning.{h,cpp}`: which widths a launch may use, and the measurement that picks one ([37](37-width-selection.md)) |

The `WarpBuilder` class and the runtime's shared state lived in anonymous namespaces; they moved to named internal namespaces (`cpusim::simt_internal`, `cpusim::rt`) declared in the internal headers. The runtime's internals are compiled with hidden visibility, so the library exports exactly the CUDA API it exported before; three internal symbols that had C linkage by accident (`is_object`, `mapped_mutex`, `mapped_ranges`) are no longer exported.

Checked: the 49 tracked programs build to binaries of the same sizes, and the 43 tests pass.

## Tests

| Test | What it pins down |
|---|---|
| `width_selection` (`tests/test_width.cpp`) | The widths a launch may use and their order, for kernels with and without barriers, missing variants, blocks smaller than a warp, refill, warps of 16; the warm-up, the drop at 3x, the median, the 10% tie rule, short launches, a measurement continued across programs; 1000 made-up cases where a clear difference must go to the faster width and a near tie must be decided the same way every time. Times are made up: no kernel runs, so the result does not depend on the machine. |
| `cuda_heavy_kernels` (`tests/cuda/heavy_kernels.cu`) | Two kernels shaped like the ones that took minutes to compile ([35](35-compile-time.md)): a loop of 32 prefix sums with barriers and an unroll pragma (sc), and 256 live floats per thread through butterflies, warp shuffles and a barrier (hadamard). Checked against a host reference at every width, and compiled under a budget of 60 s (`COMPILE_BUDGET` in `tests/cuda_feature_test.sh`): 23 s now, 106 s with the build before the two fixes. |
| `static_assert` in `src/device/device_core.h` | The VRAM range stays below the addresses where the kernel loads PIE executables ([36](36-cooperative-variant-false-sharing-and-vram.md)): the build fails otherwise. |

Measurements that cannot be a unit test, because they depend on the machine, have scripts in `bench/track` ([tutorial 6](../../tutorials/06-tracking.md)):

- `track.sh` and `compare.py`: compile time, binary size, kernel time and correctness of 49 programs, compared between two builds.
- `oracle.sh` and `oracle.py`: the automatic width against every width forced, kernel by kernel.
- `oracle.sh` with repeated automatic modes and `stability.py`: how many different widths each kernel gets from fresh daemons.
- `repeat.sh`: a program run again and again, the output of each failing run kept, for failures that come now and then.

Back to the [iteration 4 index](README.md).
