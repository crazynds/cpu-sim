# CudaSimulator

Compiles CUDA programs **without modification** to run on the CPU: each 32-thread warp executes on AVX-512 lanes, and a resident driver plays the role of the GPU (it reserves cores and "VRAM", loads modules and runs kernels). The complete design is in [DESIGN.md](DESIGN.md); the optimizations of the CPU execution are explained one by one, with theory, code and results, in [docs/optimization/](docs/optimization/README.md), separated by iteration. To run the benchmarks yourself, follow the [benchmark tutorials](docs/tutorials/README.md).

## Status

| Part | State |
|---|---|
| `cpusimd` driver + `cpu-smi` | done: VRAM with allocator, per-process contexts, memcpy via `process_vm_readv/writev`, modules, launch |
| Backends | daemon and in-process, same device interface |
| `libcpusim_cudart` | malloc/free/memcpy/memset/memGetInfo, properties, fatbin registration, `<<<>>>`, profiling |
| Compiler (`cucc` + `cpusim-devc`) | static and dynamic `__shared__`, `__syncthreads`, divergence, loops; pass on both backends: `matmul`, `transpose`, `heat2d`, `scan`, `bitonic_sort`, `mandelbrot` |
| Also | asynchronous streams (one queue per stream, streams in parallel, default stream rules, events, `cudaStreamWaitEvent`, host callbacks); warp operations (`__shfl*_sync`, `__ballot_sync`, `__any/__all_sync`, `__activemask`, `__match_*`, `__reduce_*`); managed memory and mapped host memory (shared pool at the same address in the driver and in the program); device math library (AVX-512 where available, glibc libmvec for exp/log/pow/sin/erf...); per-thread local arrays |
| Also, since the stress suite | atomics, device and `__managed__` variables, recursion and function pointers, device `printf`/`malloc`/`assert`, cooperative groups, libcu++ atomics, CUDA graphs; 629 of 631 stress cases pass ([tests/stress/](tests/stress/README.md)) |

## Build

Requires GCC 13+, CMake 3.24+ and LLVM/Clang 19. Without root, the Ubuntu packages can be unpacked into a local folder:

```sh
mkdir -p ~/.local/opt/llvm19-debs && cd ~/.local/opt/llvm19-debs
apt-get download clang-19 libclang-cpp19 libclang-common-19-dev llvm-19 llvm-19-dev \
    llvm-19-runtime llvm-19-linker-tools libclang1-19 libclang-19-dev
for d in *.deb; do dpkg -x "$d" ../llvm19; done
ln -sf /usr/lib/x86_64-linux-gnu/libLLVM.so.19.1 ../llvm19/usr/lib/x86_64-linux-gnu/libLLVM.so.19.1
```

(The system `libllvm19` must be installed; CMake looks in `~/.local/opt/llvm19` or `/usr/lib/llvm-19`.)

```sh
cmake -S . -B build
cmake --build build -j
ctest --test-dir build --output-on-failure              # everything, ~1 minute
ctest --test-dir build --output-on-failure -L features   # one test per feature, at all widths and on both backends
ctest --test-dir build --output-on-failure -L samples    # CUDA samples at test size (seconds each)
bench/run.sh both            # kernels only, GPU (nvcc) and CPU (cucc), with the result hash compared
```

## Usage

```sh
build/cpusimd --daemon               # starts the driver (config: ~/.config/cpusim/device.toml)
build/cpu-smi                        # like nvidia-smi; also -q, -L, -l 1, --query-gpu=...
build/cucc -std=c++20 -fmad=false -I samples/common samples/matmul_naive.cu -o matmul_naive
./matmul_naive                       # uses the driver if it is running; otherwise runs in-process
build/cpusimd --stop
```

In `cpu-smi`, **Dev-Util** is the fraction of time with some kernel running (what GPU-Util measures in nvidia-smi) and **CPU Load** is the real occupancy of the reserved cores, from any process. **Fan** shows the speed in RPM of the first fan. Programs with `CPUSIM_BACKEND=inprocess` (such as those of `bench/run.sh`) also appear: they publish their own status in `<socket directory>/inprocess/<pid>` and the driver adds them to the utilization, memory and processes (Type `I`). The `ctest` tests use their own driver, on a temporary socket, and do not appear in the system's `cpu-smi`.

Environment variables:

| Variable | Effect |
|---|---|
| `CPUSIM_BACKEND` | `daemon`, `inprocess` or `auto` (default) |
| `CPUSIM_PROFILE=1` | the first launch of each (kernel, grid, block) runs instrumented and counts flops, integer operations, loads/stores, branches and SIMD efficiency; the following ones run without instrumentation and are timed. Report at the end, on stderr |
| `CPUSIM_DEBUG=1` | each launch runs step by step (one warp up to the next barrier at a time) on a single thread, with the shared memory and the warp frames between guard pages. The first invalid access becomes `cudaErrorIllegalAddress` with a report: step, block, warp, address, region and instruction in the kernel module. `=2` also prints each step |
| `CPUSIM_JIT=0` | runs the generic kernels instead of code specialized for the launch's `blockDim`. By default `cucc` compiles, in parallel and ahead of time, every block shape that is a constant at a launch in the source (`k<<<grid, 128>>>`) into the binary (`--no-specialize` turns that off), and the first launch with any other shape recompiles the kernel bitcode for it (JIT), kept in `~/.cache/cpusim/jit` |
| `CPUSIM_WARP_SIZE=16` / `32` / `64` | read by `cucc` at compile time: threads per warp (`warpSize`, lane ids, ballots, shuffles, `cudaDeviceProp::warpSize`). 32 (CUDA's) by default; 16 fills exactly one AVX-512 register, for tuning code that uses `warpSize` instead of a literal 32; 64 is AMD's wavefront size (CUDA's 32-bit masks then cover lanes 0..31 only). Tensor cores need 32 |
| `CPUSIM_LANES=<n>` / `<n>s` / `<n>c` | forces the width of the thread group per call (32, 64, 128, ...; `s` = variant that suspends at barriers; `c` = cooperative, half of the block on each SMT thread of a core). Without it, the runtime chooses per kernel and block shape: in a large launch, it measures each candidate on slices of the launch's own blocks (two alternating rounds, keeping the best slice of each) and runs the rest with the winner; small launches measure one candidate per launch. The choice holds until the end of the process |
| `CPUSIM_TRIAL_BLOCKS=<n>` | blocks per worker in each slice of that measurement (default 32; short slices measure more clock and cache noise than width) |
| `CPUSIM_VERBOSE=1` | shows, per block shape, the code compiled ahead of time or by the JIT, and the width choice |
| `CPUSIM_DUMP_ASM=1` | writes to the current directory the x86 assembly (AVX-512, for the host CPU) of the launched kernels: one `cpusim-asm-<program>-<module>-<XxYxZ>.txt` per block shape (the specialized code) or `-generic.txt` with `CPUSIM_JIT=0`. The header lists each generated function (32-lane warp, wide variants, suspended ones, etc.) with the line where it starts and the instruction count; the body keeps the LLVM comments (loop depth, block names) |
| `CPUSIM_BLOCK_ORDER=linear` | disables the banded block order (swizzle) |
| `CPUSIM_CONFIG`, `CPUSIM_SOCKET`, `CPUSIM_VRAM`, `CPUSIM_CORES` | override the configuration |

`CPUSIM_PROFILE`, `CPUSIM_DEBUG` and `CPUSIM_DUMP_ASM` are read by the program, not by the driver: `libcpusim_cudart` tags each launch, and the driver (always the same one) executes that launch in the requested mode.

Configuration (`~/.config/cpusim/device.toml`, everything optional; `cpusimd --init-config` writes it with this machine's defaults, commented out):

```toml
[device]
name         = "My CPU GPU"
cores        = 12        # reserved physical cores ("SMs"); or cpus = "4-15,20-31"
vram_private = "8GiB"
hugepages    = "2MiB"    # 2MiB, 1GiB or none
prefault     = false     # true: occupies all of the VRAM in RAM at startup

[driver]
socket = "/run/user/1000/cpusim/driver.sock"
```

## Layout

```text
include/           our CUDA headers (cuda_runtime.h, vector_types.h, device side)
src/common/        config, topology, sensors, errors
src/device/        VRAM, allocator, executor, DeviceCore, backends, kernel ABI, protocol
src/driver/        cpusimd (server, utilization monitor)
src/smi/           cpu-smi
src/cudart/        libcpusim_cudart
src/compiler/      cpusim-devc: uniformity, linearization with masks, widening to 32 lanes
tools/             cucc (compile driver), device_query, memtest
share/cuda-stub/   fake SDK "installation", only so Clang picks the launch ABI
samples/           reference CUDA programs (they also run on a real GPU with nvcc)
tests/             tests (ctest)
```
