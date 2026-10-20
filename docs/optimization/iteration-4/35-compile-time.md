# 35. Compile time: kernels apart, classes of shapes, and what not to generate

> **In one sentence:** three HeCBench programs took from eight to over twenty minutes in `cucc`; each kernel now compiles in a process of its own, block shapes that produce the same code share one module, loops containing a barrier are not unrolled, and a thread with more than 64 live values gets no variant of 128 lanes or more: every tracked program compiles in under 30 s (median 1.5 s), with no kernel slower at run time.

The constraint throughout this round: run time comes first. A first attempt (commit 7436e9a) cut compile time by generating less (fewer kernels per shape, no wide variants past a size budget), and was reverted where it could cost run time; what stayed is below, each piece checked on the 49 tracked programs (`bench/track`, [tutorial 6](../../tutorials/06-tracking.md)).

## Where the time went

`cucc` compiles a `.cu` file's device code once per module: a generic module (any block shape) and one specialized module per constant block shape it finds at the launch sites ([01](../iteration-1/01-specialized-jit.md)). Each module holds, per kernel, the 32-lane warp entry and the wider variants ([02](../iteration-1/02-width-variants.md)): up to 512 lanes block-wide, and the suspending 64s and 128s for kernels with barriers. Three programs exposed what that costs:

| Program | Before | Where |
|---|---|---|
| segsort | minutes | many kernels per file, compiled one after the other in one `cpusim-devc` per module |
| sc | 499 s, then a hang at run time | a loop of 32 prefix sums with barriers, fully unrolled: 32 copies of every resume point |
| hadamard | over 1200 s | 39 kernels; the largest has 484 live floats per thread, and its 256-lane variant took 215 s alone |

A profile of `cpusim-devc` on sc and hadamard put 95% of the time in the LLVM back end, in greedy register allocation (`RAGreedy`, live interval splitting) on functions of tens of thousands of instructions.

## Kernels compiled apart

`cpusim-devc --list` prints a module's kernels, and `together` when they must stay in one process: a device function whose address is taken may be called through a pointer from any kernel, and the vector versions of such functions are built for the kernels that set the pointers up ([28](../iteration-3/28-function-calls.md)). Otherwise `cucc` starts one `cpusim-devc` per kernel per module (`--kernels <name>`), all at once, and links each module's objects into its shared object (`tools/cucc.in:296-335`):

```python
kernels, together = list_kernels(base + ".dev.bc", o)
apart = len(kernels) > 1 and not together and not o.keep
...
parts[shape] = [("%s.k%d.o" % (obj, i), "%s.k%d" % (manifest, i), ["--kernels", k]) for i, k in enumerate(kernels)]
```

Two details make the pieces link. A `__device__` variable is defined in every object that uses it, so its definition (`__cpusim_init_<name>`, `__cpusim_sym_<name>`) and the ABI version are weak symbols. And inside each process, code generation itself is parallel: the module is cut with LLVM's `SplitModule` into parts that share nothing, each part optimized and emitted in a thread with its own `LLVMContext`, the objects joined with `clang -r` (`src/compiler/main.cpp:740-810`). `linkonce` definitions (vtables, inline functions) are made `weak_odr` first, or a part's clean-up would drop one that another part uses. With function pointers the split is skipped: a table of addresses must be read where it is defined. `CPUSIM_DEVC_THREADS` shares the cores between the processes `cucc` starts.

The code is the same as one process per module would produce: the objects of the tracked programs have the same sizes.

## Classes of block shapes

A specialized module is compiled for one block shape, so that `blockDim` is a constant and the masks of a partial last group are known. For a kernel without barriers, most of that knowledge is the same across many shapes: with 1D blocks whose size is a multiple of 128, every lane of a group of 32 to 128 lanes is a thread, whatever the size. `cpusim_shape_class` (`src/device/kernel_abi.h:234-240`) names those classes:

```c
static inline uint32_t cpusim_shape_class(uint32_t x, uint32_t y, uint32_t z) {
    if (y == 1 && z == 1) {
        if (x % 128 == 0) return 1;
        if (x % 32 == 0) return 2;
    }
    return 0;
}
```

A program with blocks of 128, 256 and 512 threads now gets one module for class 1 (its kernels without barriers, `--specialize-class 1 --no-barriers`) and one module per exact shape for the kernels with barriers (`--barriers-only`), whose block-wide variants need the exact size. In a class module, `threadIdx.x` is the group's base plus the lane and `blockDim.x` a uniform value loaded from the context (`emit_class_tid`, `src/compiler/simt_ops.cpp:351-369`); the lane mask is all ones for class 1, and for class 2 at 32 lanes. The JIT uses the same key: a launch of a class shape compiles, or finds in its disk cache, the class's module for the kernel launched. A 2D class (`blockDim.y` a uniform) was tried and dropped: iso2dfd ran 20 to 30% slower without the constant row length.

Against the build before, over 3 runs each: kernel time geomean 0.995 (heat2d 0.79, bitonic-sort 0.87, zeropoint 0.90), compile time and sizes unchanged.

## Loops with a barrier stay rolled

Before the SIMT pass, kernels that still have per-thread arrays get `loop-unroll-full`, so that indices become constants and SROA puts the arrays in registers. A loop containing `__syncthreads` gained nothing from that and cost a great deal: in the warp and suspending variants, each barrier is a resume point of a state machine ([09](../iteration-1/09-state-at-barriers.md)), and every copy of the loop body adds one, with its saved state and its dispatch edge. HeCBench sc unrolled 32 rounds of a prefix sum with barriers into a function the register allocator spent 8 minutes on, and the program then timed out at run time. Such loops are now marked already unrolled before the pipeline runs (`src/compiler/passes.cpp:648-668`):

```c++
void keep_barrier_loops_rolled(Function& f) {
    DominatorTree dt(f);
    LoopInfo li(dt);
    for (Loop* l : li.getLoopsInPreorder()) {
        bool barrier = false;
        for (BasicBlock* bb : l->blocks())
            for (Instruction& i : *bb)
                if (auto* ii = dyn_cast<IntrinsicInst>(&i)) { ... barrier |= <a barrier intrinsic>; }
        if (barrier) l->setLoopAlreadyUnrolled();
    }
}
```

A barrier costs the same inside a loop as outside, so nothing is lost at run time: sc compiles in 3.3 s and its kernels take 646 ms; the other 47 tracked programs build to the same sizes.

## No 128-lane variants for heavy threads

Each varying 32-bit value of a thread takes W/16 zmm registers at W lanes, and there are 32. `kernel_features` (`src/compiler/passes.cpp`) counts, from the liveness of the per-thread code, the most 32-bit units of varying values live at once (`CPUSIM_DEVC_FEATURES=1` prints it). Collected with the runtime's width measurements over the tracked programs (`bench/track/collect_widths.sh`, `widths.py`), it bounds what the wide variants can win: the widest winner had 29 live units, and past 47 every kernel picked 32 or 64 lanes. HeCBench hadamard has kernels with 106 to 516 live units; their 256-lane and 128s variants ran 2 to 5 times slower than 32, and the largest took 215 s to compile.

A kernel with more than 64 live units now gets no variant of 128 lanes or more (`src/compiler/main.cpp:568-620`):

```c++
constexpr unsigned kWideLiveLimit = 64;
const bool heavy = kernel_features(*k.scalar).live_units > kWideLiveLimit;
...
if (!wanted(2 * L << i, true) || narrow || (heavy && (2 * L << i) >= 128)) continue;
```

The 64-lane variants stay (64s won one of hadamard's kernels). hadamard compiles in 29 s instead of 220; of the other tracked programs only interleave (2% faster) and rsbench (equal) changed.

## Results

| Program | Before | After |
|---|---|---|
| segsort | minutes | 51 s |
| sc | 499 s, hung at run time | 3.3 s, runs |
| hadamard | over 1200 s | 29 s |
| 49 tracked programs, median | | 1.5 s |
| 72 programs drawn at random ([38](38-where-we-stand.md)), median / 90th percentile | | 1.7 s / 4.4 s (nvcc: 1.3 s / 3.0 s) |

`cuda_heavy_kernels` ([39](39-organization-and-tests.md)) keeps kernels shaped like sc's and hadamard's under a compile budget of 60 s: 23 s now, 106 s with the build before the last two changes.

## Not done

- **Predicting the widths worth building.** The live units bound the wide variants from above; a model that picks one to three variants per kernel from its features would cut compile time further. The data collected (`bench/track/results/w1.features`, `w1.widths`) showed pressure alone loses 1.19 to 1.41x in geomean against measuring, so the variants are still all built.
- **frna** (22 s) and **bonds** (18 s) are the slowest of the random programs; not profiled.

Back to the [iteration 4 index](README.md).
