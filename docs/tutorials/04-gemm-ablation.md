# 4. GEMM ablation: where the `cucc` version's speed comes from

Tutorial 3 compares the `cucc` GEMM with OpenMP. This one takes the `cucc` version apart: [bench/gemm_explore.cu](../../bench/gemm_explore.cu) starts from the first version of the duel and adds one change at a time, so you can measure what each one is worth on your machine.

## Build and run

```sh
build/cucc -O3 -std=c++20 bench/gemm_explore.cu -o gemm_explore
./gemm_explore list
./gemm_explore 1-copies 6144
```

Each run prints one line:

```
<step>  n=<n>  copies <ms>  pack <ms>  kernel <ms>  total <ms> | best kernel <ms> <GFLOPS>  ok
```

- `copies`, `pack`, `kernel`, `total`: one multiplication from host data to host result, split into its phases (`copies` is upload plus download; zero with managed memory).
- `best kernel`, `GFLOPS`: the best of 3 more kernel runs.
- `ok` / `FAIL`: 16 rows of C, including the first and the last, compared exactly.

Run the steps in order with a pause between them, and repeat the series: the clock changes during it (tutorial 1), so look at differences that persist across repetitions, not at a single pair of lines.

```sh
for s in 1-copies 2-managed 3-warp-pack 4-stcs 5-kc512 6-6x64 7-kc768; do
    sleep 20; ./gemm_explore $s 6144
done
```

## The steps

| Step | Change | Why it should help |
|---|---|---|
| `1-copies` | first duel: 12×32 per warp, KC 256, packing one thread per element with 64-bit `/` and `%`, `cudaMalloc` + `cudaMemcpy` | baseline |
| `2-managed` | `cudaMallocManaged` instead of copies | on the simulator the device memory is host memory; managed memory skips two copies of n² floats in and one out |
| `3-warp-pack` | packing kernels by warps: one block per panel, each lane a contiguous column | no 64-bit division per element; contiguous loads and stores |
| `4-stcs` | packing stores with `__stcs` | non-temporal stores (`vmovntps`) write the packed copy without reading each destination line first (no read-for-ownership): a third less memory traffic in packing |
| `5-kc512` | KC 256 → 512 | C is read and written once per pass of k: half as many passes |
| `6-6x64` | micro-kernel 12 rows × 32 threads → 6 rows × 64 threads | both keep 24 zmm accumulators, but per step of k the 12×32 tile does 2 vector loads of B and 12 broadcasts of A (14 memory operations) for 24 FMAs, and 6×64 does 4 + 6 = 10. Memory operations cost power, and power sets the clock (tutorial 2) |
| `7-kc768` | KC 768 | fewer passes over C again; the kernel the duel uses |
| `8-mc24` | 24 rows per block instead of 96 | more, smaller blocks: pays when the grid of 96-row blocks is too small to keep every multiprocessor busy to the end. Compare with `7-kc768` at the smallest size |

Alternatives to compare with `7-kc768`: `x-12x32-kc768` (the old tile with the new KC), `x-kc256`, `x-nc256` (256 columns per block).

## What to look for

- **Where the total goes.** The `copies`, `pack` and `kernel` columns of step 1 show what the first version spent its time on, and steps 2 to 4 remove the first two.
- **The kernel against the peak.** Run `./peak fp32` (tutorial 2) before the series, and note `clocks.sm` from `cpu-smi` during steps 5 to 7. How close the kernel gets to the peak at its own clock tells whether a step saved cycles or saved power.
- **`__stcs` depends on the cache.** Streaming stores send the packed copy to memory; when the copies (8n² bytes for both) fit in the last-level cache, that pushes them out right before the kernel reads them. Compare `3-warp-pack` and `4-stcs` at a size whose copies fit in your cache (`cpu-smi` does not show it; `lscpu | grep L3`) and at one whose copies do not. This is why the duel version decides from `l2CacheSize`.
- **Every line must say `ok`.**

## Checking that the streaming stores are real

Dump the generated code and look for `vmovntps` and `sfence` in the packing kernels:

```sh
mkdir -p /tmp/asm && (cd /tmp/asm && CPUSIM_DUMP_ASM=1 "$OLDPWD/gemm_explore" 4-stcs 1536) && grep -c vmovntps /tmp/asm/cpusim-asm-*.txt
```

(`CPUSIM_DUMP_ASM` writes one listing per block shape into the current directory.) The `sfence` at the end of those kernels is required: non-temporal stores are weakly ordered, and the next kernel reads the packed data from other cores. `tests/cuda/streaming.cu` checks that every store path is correct.

## Ideas that did not pay off

These are not in the program, but are easy to try by editing it:

- **Reading A without packing it**: when a row of A (4n bytes) is a multiple of 4 KiB, as for n = 6144, the rows one micro-tile reads map to the same L1 sets and evict each other.
- **Copying A into `__shared__` inside the kernel**: removes the `pack_a` pass, but costs about as much inside the kernel.
- **A 14×32 tile** (28 accumulators): needs edge handling for the rows, for little gain.
