# Iteration 3

The third round: it started with stability tests that require the same bits in all forms of execution (and found a real rounding difference), added the 16-bit floating-point types and the tensor cores (`nvcuda::wmma`) on top of the Zen 4 dot-product instructions, and fixed along the way three compiler and runtime problems that this work exposed. It ended with a GEMM written for the CPU, not for the GPU, near the Zen 4 theoretical peak, which led the bfloat16 `mma_sync` to be emitted inline.

| # | Document | In one line |
|---|---|---|
| 23 | [Reproducibility and stability](23-reproducibility-and-stability.md) | The same bits in all execution modes, and FMA contraction decided in the front end |
| 24 | [16-bit types](24-16-bit-types.md) | `__half` and `__nv_bfloat16` computed in `float` and rounded only once |
| 25 | [Tensor cores](25-tensor-cores.md) | `wmma::mma_sync` becomes one call per warp to `vdpbf16ps`, `vpdpbusd` or F16C with FMA |
| 26 | [Compiler and runtime changes](26-compiler-and-runtime-changes.md) | Division by a constant in divergent code, the generic-variant budget and the JIT cache key |
| 27 | [A kernel written for the CPU, near peak](27-kernel-at-peak.md) | GEMM in CUDA for the `cucc` model: 94% (fp32) and 93% (bf16) of one core's peak, with the bfloat16 `mma_sync` inline and the accumulators in registers |
| 28 | [Function calls that are not inlined](28-function-calls.md) | Recursion, function pointers and virtual calls: a W-lane version of the function, called with the warp's mask |
| 29 | [The stress suite](29-stress-suite.md) | 631 cases checked against the GPU: from 174 passing to 629, what broke and why, and the slow cases left for later |
| 30 | [Launches and the slow cases](30-launches-and-hot-spots.md) | An empty kernel from 42 us to 1.7 us; atomics merged per warp; against OpenMP, 335 of 541 cases faster (233 before) |
| 31 | [Micro-optimizations and 32-bit gathers](31-gathers-and-scatters.md) | Instruction order and masks do not matter with SMT; addresses recomputed where used and 32-bit gather indices: bitonic sort 1.18x |
| 32 | [The warp size](32-warp-size.md) | `CPUSIM_WARP_SIZE=16/32/64`; blocks smaller than a warp in fewer lanes (10 to 46% faster) |
| 33 | [A two-socket Xeon](33-xeon-server.md) | H100, OpenMP and the simulator on 2 x 32 cores; launches 816 to 3.7 us (monitor, wake-ups, workers per job, C-states) |
| 34 | [Adapting to the CPU](34-adapting-to-the-cpu.md) | Code for this CPU only; native fp16 fma/sqrt; AMX tensor path (opt-in); small launches on one socket |

## Tensor core results at the end of the iteration

`samples/tensor_gemm.cu`, the same source on both sides; GPU = RTX 5060 Laptop with `nvcc`; cucc = our compiler, on 28 CPUs.

| Size | Type | cucc | GPU |
|---|---|---|---|
| 2048^3 | bf16 | 0.62 TFLOPS | 15.34 TFLOPS |
| 2048^3 | fp16 | 0.36 TFLOPS | 17.63 TFLOPS |
| 2048^3 | int8 | 0.77 TOPS | 16.46 TOPS |
| 4096^3 | bf16 | 0.55 TFLOPS | 14.06 TFLOPS |
| 4096^3 | fp16 | 0.36 TFLOPS | 15.74 TFLOPS |
| 4096^3 | int8 | 0.59 TOPS | 16.26 TOPS |

After the inline bfloat16 path ([27](27-kernel-at-peak.md)), bf16 at 2048^3 went to 1.06 TFLOPS (in the same run, fp16 0.80 TFLOPS and int8 1.67 TOPS, types that still use the library). `samples/cpu_gemm.cu`, written for the CPU, reaches 3.17 TFLOPS in bf16 and 1.64 TFLOPS in fp32 with all cores.

## Open points

Left by iteration 2:

- **Block distribution policy:** chunks of 64 blocks helped the stencils and cost ~5% of cycles in matmul; it could become one more dimension of the automatic selection (see [17](../iteration-2/17-block-distribution.md)).
- **Measurement noise:** the same configuration varied by up to ±30% between runs in black-scholes, which limits the precision of the width choice (see [03](../iteration-2/03-automatic-width-selection.md)).
- **Missing compiler features:** atomics, device `malloc` and `__managed__` variables (all done in [29](29-stress-suite.md)).

New in this iteration:

- **64-bit integers above 2^53 to bfloat16:** the conversion goes through `double`, which already rounds; it may get the tie-break wrong (see [24](24-16-bit-types.md)).
- **Tensor cores:** only the 16x16x16 shape; m32n8k16, m8n32k16, tf32, `double` and `satf` for integer accumulators are missing. fp16 falls below bf16 because Zen 4 has no half dot product (see [25](25-tensor-cores.md)).
- **fp16 and int8 without the inline path:** only bfloat16 with a `float` accumulator has the `mma_sync` emitted inline with accumulators in registers; half, int8 and uint8 still make one library call per `mma` (see [27](27-kernel-at-peak.md)).
- **2-byte fragment loads:** they become two masked 32-byte loads and a `vinserti64x4`; the two 16-lane halves are not merged into one 64-byte load (see [27](27-kernel-at-peak.md)).
- **Efficiency with all cores depends on temperature:** the same bf16 GEMM went from 1956 to 3173 GFLOPS depending on the laptop's thermal state; the loss that remains relative to the peak was not attributed (see [27](27-kernel-at-peak.md)).
- **Compile time of large kernels:** the generic-variant budget resolves the module compiled before execution, but the JIT of a WMMA kernel for 512-thread blocks still takes 7.43 s, and the 16-bit types test compiles in 13.6 s, almost all of it in register allocation (see [26](26-compiler-and-runtime-changes.md)).

New in the stress suite ([29](29-stress-suite.md)):

- **Slow cases** (after [30](30-launches-and-hot-spots.md)): 31 of the 114 cases whose OpenMP version takes 1 ms or more still run slower on the simulator: atomics contended across cores, GPU-style GEMM and SpMV kernels, per-thread divergence, byte-per-thread code; and a launch through the driver costs 5 to 7 us against 1.7 in-process.
- **Waiting loops:** a kernel that waits for other threads runs one warp per call; a lane waiting for another lane of its own warp still hangs.
- **`j0` and `y0`:** glibc's double versions exceed CUDA's ulp bound near their zeros.

Previous: [iteration 2](../iteration-2/README.md). Next: [iteration 4](../iteration-4/README.md). Back to the [index](../README.md).
