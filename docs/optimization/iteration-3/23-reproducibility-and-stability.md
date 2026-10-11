# 23. Reproducibility and stability

> **In one sentence:** a new set of tests (label `stability` in CTest) requires the kernels to produce the same bits in every way of executing that the runtime can choose (widths, JIT or generic module, number of cores, block order, backend), requires every thread to run exactly once in any launch geometry, and requires that switching widths during the automatic choice does not change the result; on its first run, it found a real rounding difference caused by the contraction of `a*b+c` into FMA, fixed by replacing `-ffp-contract=fast` with `-ffp-contract=on` in `cucc`.

The general concepts (width variants, generic module, JIT, executor) are in [00-basic-concepts.md](../00-basic-concepts.md).

## The problem

The same kernel can be executed in many different ways, and the runtime chooses among them on its own:

- **variant width:** 32, 64, 128, 256 or 512 lanes per call, with the forms that suspend at barriers (`64s`, `128s`) and the SMT cooperative ones (`128c`, `256c`) ([02](../iteration-1/02-width-variants.md), [15](../iteration-2/15-smt-cooperative-variant.md));
- **automatic choice:** in the first launches, slices of blocks of the same launch run with different widths until the runtime makes up its mind ([03](../iteration-2/03-automatic-width-selection.md));
- **JIT or generic module:** the code specialized for the block shape ([01](../iteration-1/01-specialized-jit.md)) or the one compiled before execution (`CPUSIM_JIT=0`);
- **distribution:** number of cores, block order ([04](../iteration-1/04-block-order.md)), splitting of blocks into warps ([16](../iteration-2/16-splitting-blocks-into-warps.md)), barrier pass ([10](../iteration-2/10-barrier-pass.md));
- **backend:** the in-process executor or the `cpusimd` daemon.

The functionality tests (`tests/cuda_feature_test.sh`) already ran each program in 12 of these modes, but each program only compared the result with a reference computed on the host, generally with a relative tolerance (`close_rel`) for floating point. A difference of 1 ulp between two modes went unnoticed. On the GPU, the same binary with the same input gives the same bits on every execution, whenever the kernel does not depend on the order between threads (floating-point atomics, for example). A program that compares its output with a reference file, or that iterates until convergence, may behave differently when the chosen width changes from one execution to another. Systematic coverage was also missing for unusual launch geometries, divergent control flow with nested loops, and repeated launches during the automatic choice.

## The idea

CUDA semantics fixes every rounding of many kernels: per-thread operations, reductions in shared memory or by `__shfl_xor_sync` in a fixed order, stencils and local arrays. For these kernels, the only correct output is one sequence of bits, and any difference between modes is a defect, not numerical noise. So the test does not need a tolerance: a hash of the output bits, compared between executions, is enough.

Two separate properties are verified:

1. **reproducibility:** the hashes are equal in all modes and in repeated executions of the same mode;
2. **correctness:** the values match a model on the host, so that a stable but wrong result also fails.

In addition, the compiler itself must be deterministic: two compilations of the same source must generate byte-for-byte identical executables (no iteration order over pointers, temporary names or dates in the binary).

## How it works, step by step

### 1. The hashes

`tests/cuda/repro.cu` has 5 kernels chosen to exercise different compiler paths, and prints 6 lines `DIGEST <name> <hash>` (the first kernel produces a `float` output and a `double` one):

| Kernel | What it exercises |
|---|---|
| `math_mix` | libmvec functions (`sinf`, `expf`, `powf`, `atan2f`...), IEEE division and square root, `fmaf`, and the same computations in `double` |
| `block_reduce` | tree reduction in dynamic shared memory, with `__syncthreads` |
| `warp_reduce` | butterfly reduction with `__shfl_xor_sync` |
| `stencil` | 5-point stencil on a 2D grid of 16x8 blocks, with divergent borders and partial blocks on both axes |
| `local_sort` | local array indexed by data (insertion sort) and weighted sum |

The hash is 64-bit FNV-1a over the bytes of the output, with one exception: NaNs are replaced by a canonical NaN beforehand, because the payload of a NaN is not specified by IEEE 754 for most operations and is not part of the result.

### 2. The 19 executions

`tests/cuda_repro_test.sh` compiles the program twice and compares the executables with `cmp`; then it runs the program 19 times:

```text
auto, auto2, auto3                 automatic choice, 3 times
lanes-32 ... lanes-512             the 9 forced widths (CPUSIM_LANES)
no-jit                             generic module (CPUSIM_JIT=0)
one-core, three-cores              CPUSIM_CORES=1 and 3
linear-order                       CPUSIM_BLOCK_ORDER=linear
no-barrier-pass                    CPUSIM_NO_BARRIER_PASS=1
tiny-trials                        CPUSIM_TRIAL_BLOCKS=1 (slices of 1 block per worker in the choice)
daemon                             cpusimd backend
```

Each execution records its `DIGEST` lines, and all are compared with those of the first. `CPUSIM_TRIAL_BLOCKS=1` is the most aggressive case of the automatic choice: the measurement slices have one block per worker, so a single launch mixes the largest number of widths.

### 3. The other three tests

They run through `tests/cuda_feature_test.sh`, in the same modes as the functionality tests (10 widths, generic module and daemon):

- **`tests/cuda/launch_shapes.cu`:** 15 geometries, from `<<<1, 1>>>` to blocks of 1024 threads, 2D and 3D grids, blocks with `x` not a multiple of 32 (partial last warp), blocks only in `z` (`dim3(1, 1, 64)`) and grids of 20000 blocks. Each thread adds 1 at a position of its own (`hits[g] += 1`), and the test requires `hits == 1`: a thread that ran twice or not at all shows up. It also checks the built-ins (`threadIdx`, `blockIdx`, `blockDim`, `gridDim`) and a sum through dynamic shared memory after a `__syncthreads`. Each geometry runs with and without a barrier, because the kernel without a barrier may take the path of splitting blocks into warps.
- **`tests/cuda/control_flow.cu`:** the functions are `__host__ __device__`, so the reference is the same code executed on the host, with an exact integer result. It covers loops with a per-lane iteration count (Collatz with a limit), `break` and `continue` inside `switch`, a nested variable loop, early return at several points, and a uniform loop with barriers around divergent regions in which some threads do nothing.
- **`tests/cuda/repeat_launch.cu`:** 24 launches of three interleaved configurations (a kernel without a barrier, one with a barrier in blocks of 256 and the same one in blocks of 64, which is a different width choice). Each launch is compared with `memcmp` to the first of its configuration, while the runtime switches width underneath.

## The discovery: FMA contraction depended on the shape of the code

On the first run, `stencil` gave different bits with `CPUSIM_JIT=0` than in all the other modes. The computation is:

```cpp
out[y * w + x] = 0.6f * c + 0.1f * (in[y * w + x - 1] + in[y * w + x + 1] + in[(y - 1) * w + x] + in[(y + 1) * w + x]);
```

and there are two ways to contract `a*b + c` into an FMA here (`fma(0.6f, c, 0.1f * s)` or `fma(0.1f, s, 0.6f * c)`), besides not contracting. Each one rounds at a different point: the FMA computes `a*b + c` with a single rounding, so the product that enters it is not rounded, whereas the other product is. All are permitted by CUDA with `-fmad=true` (the `nvcc` default), but a given binary has to choose one.

`cucc` passed `-ffp-contract=fast` to Clang. What each value does:

- **`off`:** never contracts.
- **`on`:** contracts only within one source expression (the rule of `#pragma STDC FP_CONTRACT ON` from C). Clang does this in the front end: it emits `a*b + c` as the `llvm.fmuladd` intrinsic, whose semantics are "fused or not, at the discretion of the target". On x86 with FMA, instruction selection always turns it into `vfmadd`, because the target declares FMA faster than `vmulps` + `vaddps`.
- **`fast`:** the front end emits separate `fmul` and `fadd`, with the `contract` flag, and the backend decides where to form FMA: the `DAGCombiner`, which works on the selection graph of one basic block at a time and applies heuristics that depend on the shape of that graph (which values are in the same block, how many uses each product has).

The code specialized by the JIT has a different shape: with a constant `blockDim`, the border conditions and the index computations simplify, the basic blocks change and the combiner now sees a different graph. With `fast`, the decision of where to fuse changed along with it. With `on`, the decision is made in the front end, before any specialization or variant, and is recorded in the bitcode as `llvm.fmuladd`; all the variants start from the same bitcode and arrive at the same FMAs.

Two confirmations of the cause: with `-fmad=false` (which becomes `-ffp-contract=off`, with no FMA at all) all the modes already gave the same bits; and with `on` as well.

## The code

### The flag in `cucc`

`tools/cucc.in:157-163`

```python
def compile_cu(src, obj_out, o, tmp):
    base = os.path.join(tmp, os.path.splitext(os.path.basename(src))[0])
    # "on" fuses a*b+c into fmuladd in the front end, per expression, before the
    # kernel is specialized; with "fast" the backend decides per basic block, so
    # the JIT and generic variants could round differently (tests/cuda/repro.cu).
    contract = ["-ffp-contract=on" if o.fmad else "-ffp-contract=off"]
    common = CUDA_FLAGS + [o.std] + o.cpp + contract
```

`o.fmad` comes from `-fmad=true|false`, the same option as in `nvcc`, with `true` as the default. The flag goes to both compilations (device and host), so the `__host__ __device__` functions also follow the same rule on both sides.

### The hash

`tests/cuda/repro.cu:13-32`

```cpp
// FNV-1a over the raw bits; NaNs are canonical so payload bits do not count.
static uint64_t digest(const void* p, size_t bytes, size_t elem = 4) {
    const auto* b = static_cast<const unsigned char*>(p);
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < bytes; i += elem) {
        unsigned char tmp[8];
        std::memcpy(tmp, b + i, elem);
        if (elem == 4) {
            float f;
            std::memcpy(&f, tmp, 4);
            if (std::isnan(f)) std::memcpy(tmp, "\x00\x00\xc0\x7f", 4);
        } else {
            double d;
            std::memcpy(&d, tmp, 8);
            if (std::isnan(d)) std::memcpy(tmp, "\x00\x00\x00\x00\x00\x00\xf8\x7f", 8);
        }
        for (size_t k = 0; k < elem; ++k) h = (h ^ tmp[k]) * 1099511628211ull;
    }
    return h;
}
```

The bytes `00 00 c0 7f` are, in little-endian, `0x7fc00000`, the canonical quiet NaN of `float`; `0x7ff8000000000000` is that of `double`.

### Compiler determinism and the executions

`tests/cuda_repro_test.sh:14-20`

```sh
"$CUCC" -std=c++20 -I "$(dirname "$SRC")" "$@" "$SRC" -o "$DIR/$NAME"
"$CUCC" -std=c++20 -I "$(dirname "$SRC")" "$@" "$SRC" -o "$DIR/$NAME.again"
if cmp -s "$DIR/$NAME" "$DIR/$NAME.again"; then
    echo "compiler determinism: PASS"
else
    echo "compiler determinism: FAIL (two compilations of $SRC differ)"
fi
```

`tests/cuda_repro_test.sh:31-40`

```sh
run auto
run auto2
run auto3
for lanes in 32 64 64s 128 128s 128c 256 256c 512; do run "lanes-$lanes" CPUSIM_LANES=$lanes; done
run no-jit CPUSIM_JIT=0
run one-core CPUSIM_CORES=1
run three-cores CPUSIM_CORES=3
run linear-order CPUSIM_BLOCK_ORDER=linear
run no-barrier-pass CPUSIM_NO_BARRIER_PASS=1
run tiny-trials CPUSIM_TRIAL_BLOCKS=1
```

After that comes the execution on the daemon, and a loop compares each `digests.*` file with `digests.auto`; any difference prints the `diff` and the result `FAIL`.

### One thread, one execution

`tests/cuda/launch_shapes.cu:13-28`

```cpp
__global__ void shapes(unsigned* ids, unsigned* sums, unsigned* hits) {
    extern __shared__ unsigned s[];
    const unsigned tid = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
    const unsigned threads = blockDim.x * blockDim.y * blockDim.z;
    const unsigned block = blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);
    const size_t g = size_t(block) * threads + tid;
    ids[g] = (threadIdx.x | threadIdx.y << 10 | threadIdx.z << 20) ^ (block * 2654435761u);
    hits[g] += 1;  // g is unique to the thread: 2 means it ran twice
    s[tid] = tid + block;
    __syncthreads();
    if (tid == 0) {
        unsigned sum = 0;
        for (unsigned t = 0; t < threads; ++t) sum += s[t];
        sums[block] = sum ^ (blockDim.x << 1) ^ (gridDim.y << 7);
    }
}
```

`hits` is zeroed before each launch. An inactive lane that wrote anyway (wrong mask in the partial last warp), a block executed by two workers or a warp forgotten in the block splitting show up as `hits[g] != 1`. `ids` encodes the three `threadIdx` coordinates and the linear index of the block, so a swapped built-in also shows up.

### The test during the automatic choice

`tests/cuda/repeat_launch.cu:17-26`

```cpp
__global__ void tiled_sum(const float* x, float* out, int n) {
    __shared__ float tile[256];
    int t = threadIdx.x, i = blockIdx.x * blockDim.x + t;
    tile[t] = i < n ? x[i] : 0.0f;
    __syncthreads();
    float acc = 0.0f;
    for (int k = 0; k < (int)blockDim.x; ++k) acc += tile[(t + k) % blockDim.x] * (k + 1);
    __syncthreads();
    if (i < n) out[i] = acc;
}
```

The sum of `blockDim.x` terms is sequential per thread, in a fixed order, so the result has a single correct value in bits, whatever the variant. It is the kernel with a barrier that switches width the most during the choice.

### The CTest registration

`CMakeLists.txt:141-159`

```cmake
    # Compares durations (concurrent streams): alone, or other tests' load
    # decides the result.
    set_tests_properties(cuda_async PROPERTIES RUN_SERIAL TRUE)

    # Stability: launch geometry, divergent control flow and repeated launches
    # under the width tuning, in every mode as above; then bitwise
    # reproducibility of the results across all modes and runs, and of the
    # compiler output.
    foreach(feature IN ITEMS launch_shapes control_flow repeat_launch)
        add_test(NAME stability_${feature}
            COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tests/cuda_feature_test.sh ${CMAKE_BINARY_DIR}/cucc
                    $<TARGET_FILE:cpusimd> ${CMAKE_CURRENT_SOURCE_DIR}/tests/cuda/${feature}.cu)
        set_tests_properties(stability_${feature} PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL" LABELS stability TIMEOUT 120)
    endforeach()
    add_test(NAME stability_repro
        COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tests/cuda_repro_test.sh ${CMAKE_BINARY_DIR}/cucc
                $<TARGET_FILE:cpusimd> ${CMAKE_CURRENT_SOURCE_DIR}/tests/cuda/repro.cu)
    set_tests_properties(stability_repro PROPERTIES PASS_REGULAR_EXPRESSION "repro reproducibility: PASS"
                                                    FAIL_REGULAR_EXPRESSION "FAIL" LABELS stability TIMEOUT 120)
```

`ctest -L stability` runs only these four. `stability_repro` requires the `PASS` line in addition to having no `FAIL`, so that a script that dies midway does not pass by omission.

`cuda_async` now has `RUN_SERIAL`: it compares durations of concurrent streams, and under the load of the other tests running in parallel in `ctest -j` it failed once because of that load.

## When it applies (and when it does not)

The bit-by-bit comparison holds for kernels whose result CUDA fully determines:

- per-thread operations, including math functions (libmvec is the same in all variants);
- reductions in a fixed order (tree in shared memory, butterfly of shuffles);
- local arrays, stencils, loops with a per-lane iteration count.

It does not hold for kernels whose result depends on the order between threads or blocks, which also varies on the GPU between executions: floating-point `atomicAdd`, reductions that use arrival order (atomic counters that pick the last block), or reads of memory written by other blocks without synchronization. These are left out of `repro.cu` on purpose.

The contraction rule with `on` also does not make the result equal to the GPU's: `nvcc` has its own policy for where to form FMA, and a program compiled by the two may differ by 1 ulp at the points where the choice differs. What it guarantees is that, in `cucc`, the choice is the same in all the variants, JIT and generic module.

## Results

- **Test time:** `stability_repro` 8.8 s, `stability_launch_shapes` 3.4 s, `stability_control_flow` 5.6 s, `stability_repeat_launch` 3.8 s.
- **Automatic choice during `repeat_launch`:** `tiled_sum` in blocks of 256 went through the widths 256, 128c, 128s, 64s and 32 and ended up with 32; the 24 launches gave the same bits as the first.
- **Defect found:** `stencil` with `CPUSIM_JIT=0` differed from the other modes; fixed with `-ffp-contract=on`. After that, all the hashes match across the 19 executions and the two compilations generate the same executable.
- **Cost of `on` instead of `fast`:** within measurement noise.

| Benchmark | `fast` | `on` |
|---|---|---|
| HeCBench nbody | ~889 GFLOP/s | ~893 GFLOP/s |
| HeCBench black-scholes | ~49.4 ms | ~49.4 ms |
| matmul_naive | 173 ms | 176 ms |

## Limitations and trade-offs

- `on` contracts less than `fast`: an `a*b` in one statement and the `+ c` in another (`t = a * b; r = t + c;`) do not become FMA. In the measured benchmarks this did not change the time, but a kernel written in that style may lose FMAs.
- The tested modes are the ones that exist today. A new variant needs to be added to the script's list to be covered.
- `repro.cu` covers a sample of code patterns. A pattern that is not in it (for example, a reduction with `__shfl_down_sync` and a partial mask) is only covered by the functionality tests, with tolerance.

## See also

- [00-basic-concepts.md](../00-basic-concepts.md): variants, generic module and JIT.
- [01-specialized-jit.md](../iteration-1/01-specialized-jit.md): where the difference in shape between the JIT code and the generic code comes from.
- [03-automatic-width-selection.md](../iteration-2/03-automatic-width-selection.md): the measurement slices that `repeat_launch` and `CPUSIM_TRIAL_BLOCKS=1` exercise.
- [11-llvm-tuning.md](../iteration-1/11-llvm-tuning.md): other options passed to Clang and LLVM.
- [24-16-bit-types.md](24-16-bit-types.md): the types test, which runs in the same modes.
