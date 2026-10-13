# 34. Adapting to the CPU: its extensions, decided when compiling

> **In one sentence:** the code `cucc` generates now targets the extensions the CPU really reports and refuses to run on another kind of CPU (as a CUDA binary without code for its GPU); with AVX512-FP16 the half-precision fma and square root are native instructions; AMX runs bf16 tensor cores correctly but is slower than `vdpbf16ps` until the accumulator stays in a tile (opt-in); and on a two-socket machine a small launch takes its workers from one socket.

The machine and the measurements of the previous round are in [33](33-xeon-server.md). The rule throughout: every decision about the CPU is made when compiling (the compiler for this machine, or the JIT for a block shape) or when a module loads, never on the launch path.

## Compiled for this CPU, run on this kind of CPU

`cucc`'s default target is the machine it runs on (`--mcpu=native`). Until now that meant the CPU model's name (`sapphirerapids`), with the extensions LLVM assumes for that model, which a virtual machine or a BIOS setting may not provide. `cpusim-devc` now passes the extensions the CPU reports (`sys::getHostCPUFeatures`) and writes a signature of the CPU in the module's manifest (`cpu=<hash>`): a hash of the CPUID leaves that list extensions and of XCR0, the register state the OS enables (AVX-512, AMX), in `src/common/cpu_id.h`. When the runtime loads a module, it compares the signature with the CPU's. On another kind of CPU the kernels do not launch: every launch returns `cudaErrorNoKernelImageForDevice`, with one message saying to rebuild. A program built on the Xeon (AVX512-FP16, AMX) would otherwise have died with `SIGILL` on a Zen 4 at the first half-precision instruction. The JIT's disk cache includes the signature too, so a home directory shared by two machines never hands one the other's code. `--mcpu=<cpu>` (no signature) builds for a named CPU and runs wherever it is accepted.

## Half precision

`__half` arithmetic is written as `float` operations rounded once to half. LLVM already narrows `fptrunc(fadd(fpext a, fpext b))` to a half addition (likewise subtraction, multiplication and division), and with AVX512-FP16 those became `vaddph`, `vmulph`, `vdivph` (checked in the assembly). Two did not:

- **`__hfma`** is a sum rounded to odd in double, then rounded to half: one rounding, as on the GPU, but a dozen double instructions. The header now calls a function with a known name (`__cpusim_hfma_half`) holding that emulation; when the target has AVX512-FP16, `cpusim-devc` replaces the calls with `llvm.fma.f16`, a `vfmadd*ph`, which rounds once too. Elsewhere the function is inlined as before.
- **`hsqrt`** is a float square root rounded to half; float has more than twice half's precision, so `fptrunc(sqrt(fpext h))` is exactly the half square root, and `cpusim-devc` rewrites it as one (`vsqrtph` with AVX512-FP16, the same promotion elsewhere).

`type_half` (every bit pattern of the conversions, the arithmetic, a 64-step fma accumulation) passes bit for bit on the Xeon.

## AMX for tensor cores

One `tdpbf16ps` multiplies a 16 x 32 bf16 tile by a 32 x 16 one into a 16 x 16 float tile. A microbenchmark on the Xeon, per 16 x 16 x 16 `mma_sync`:

| | Cycles |
|---|---|
| `tdpbf16ps`, accumulator in a tile | 13.4 |
| load C, A, B into tiles, `tdpbf16ps`, store C | 76 |
| 128 `vdpbf16ps` (the current code), estimated from the GEMM | ~70 to 230 |

With the target's `amx-bf16`, `emit_mmav` emits the tile instructions inline (LLVM inserts the tile configuration at the function's entry): A's fragment, for a 32-lane warp, is its tile row by row (32 bytes apart); B's pairs, already interleaved for `vdpbf16ps`, are the VNNI rows the tile operand wants; the accumulator goes through the scratch area. The executor asks the OS for the tile state once (`arch_prctl(ARCH_REQ_XCOMP_PERM)`). The result is correct, and slower: 4.6 TFLOPS against 5.4 on the 4096^3 bf16 GEMM, because the accumulator moves between zmm registers and the tile around every `mma_sync`. It pays only with the accumulator in a tile through the whole loop over k, which means turning the accumulator's phis into `x86_amx` values (LLVM's AMX lowering can merge casts through phis of one vector, `optimizeAMXCastFromPhi`; ours are eight phis of W-lane vectors). Until then it is opt-in: `CPUSIM_AMX=1` for `cucc`.

## Two 512-bit FMA units

The Xeon has two FMA units of 512 bits per core where Zen 4 has two of 256. The width selection measures the variants on the machine itself in the first launches, and it picks what it picked on the laptop: 32 lanes for mandelbrot (its loop diverges: 64 lanes cost 40% more), 64 for the naive matmul. The GEMM written for warps of 16 reaches 90% of one core's peak here, against 75% for the warp-32 one ([33](33-xeon-server.md)): with two FMA units, a micro-tile with more columns per thread (fewer loads per FMA) pays. A latency-bound loop such as mandelbrot's would need more independent chains per core without more divergence; not done.

## A small launch on one socket

On two sockets, every atomic operation of joining and leaving a job is several times dearer when its workers are on both: a 1024-block kernel took 120 us with all 124 workers, 40 with one socket's. The executor now keeps, per NUMA node (read from `/sys`), the demand slots, the count of spinning workers and the condition variable its idle workers sleep on. A job whose workers fit in one node takes them from the node with the most workers awake (usually the one that ran the previous launch, whose caches hold its data; taking nodes in turn moved a small kernel's data across sockets every launch); a larger one posts to every node its share. One node on a single-socket machine: nothing changes there.

| Xeon, in-process | Before | After |
|---|---|---|
| 1024 blocks of 256 | 105 us | 85 to 89 us |
| `smith_waterman_diag_launches` (2047 launches) | 36.8 ms | 27.9 ms |
| `lcs_tiled_wavefront` | 19.5 ms | 14.5 ms |
| `floyd_warshall_blocked_shared` | 11.0 ms | 9.4 ms |

## Not done

- **C-states from the driver.** Keeping the reserved CPUs out of C6 while kernels run (`/dev/cpu_dma_latency`, per-CPU PM QoS) needs root; the driver runs as the user. Disable C6 in the machine's configuration (`cpupower idle-set -D 10`) on a server.
- **AMX with the accumulator in a tile**, above.
- **AMX for int8** (`tdpbssd`), and for fp16 on CPUs with AMX-FP16.
- **Launches of large grids** on two sockets still cost ~85 us: a per-node join counter (each node's workers on their own line) would remove most of the cross-socket traffic left.

Previous: [33](33-xeon-server.md). Back to the [iteration 3 index](README.md).
