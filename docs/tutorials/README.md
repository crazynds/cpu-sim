# Benchmark tutorials

Step-by-step instructions to run, on your own machine, every benchmark in this repository: the arithmetic peak of the device, the fp32 GEMM duel (cuBLAS on a GPU, OpenMP, and CUDA compiled by `cucc`), the ablation that explains where the `cucc` version's speed comes from, and the tensor core and sample benchmarks.

Each tutorial lists the commands, what the output means, and how to check that a claim holds rather than trusting a number.

| # | Tutorial | What it covers |
|---|---|---|
| 1 | [Setup and measurement hygiene](01-setup.md) | CPU requirements, build, tests, the driver, `cpu-smi`, and how to get stable numbers |
| 2 | [Arithmetic peak and clock](02-peak.md) | the ceiling: GFLOPS the device can do at the clock it holds |
| 3 | [The fp32 GEMM duel](03-gemm-duel.md) | GPU vs OpenMP vs `cucc`: kernel time, total time, GFLOPS |
| 4 | [GEMM ablation](04-gemm-ablation.md) | the effect of each change in the `cucc` version, one at a time |
| 5 | [Tensor cores and the other benchmarks](05-tensor-cores-and-samples.md) | WMMA and capability gating, every tensor operation (`tensor_ops`), `cpu_gemm`, `tensor_gemm`, `bench/run.sh`, the samples |

Do them in order: tutorial 1 builds everything the others use, and tutorial 2 gives the reference that tutorials 3 to 5 are compared against.

The programs these tutorials use:

| Program | Purpose |
|---|---|
| `bench/peak.cu` | fp32 FMA and bf16 tensor peak of the device, through CUDA (also builds with `nvcc`) |
| `build/cpu-smi` | the device's state while a benchmark runs: clock, temperature, utilization (like `nvidia-smi`) |
| `bench/gemm_fp32.cu` | the duel program (one source, built with `nvcc` for the GPU and with `cucc` for the CPU) |
| `bench/gemm_duel.sh` | builds and runs the duel in pairs with pauses, then prints the means and ratios |
| `bench/gemm_explore.cu` | the ablation steps of the `cucc` GEMM |
| `bench/tensor_ops.cu` | every tensor core operation (type and shape): WMMA on the GPU or the simulator, or the same product with OpenMP |
| `bench/run.sh` | the earlier kernel comparison (GPU, `cucc`, plain C++), see [BENCHMARKS.md](../../BENCHMARKS.md) |
