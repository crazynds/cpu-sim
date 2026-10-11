# 21. Compile time

> **In one sentence:** four changes took the project out of a seven-minute test suite and a kernel that never finished compiling: tests with small workloads and parallel references, tree-shaped joining of vector chunks, a workaround for an LLVM 19 bug that hung code generation, and a leaner rare path for scattered loads at large widths.

## The problem

This document is different from the others: it is not about kernel speed, but about the time spent **before** running them, compiling and testing. This matters because each optimization experiment requires recompiling and running the entire suite; if it takes seven minutes, each idea costs seven minutes to verify.

Four problems appeared, one after another:

1. **The sample tests took 438 s**, of which 232 s were mandelbrot alone. The samples (example programs in `samples/`) were run with GPU-sized workloads, meant to keep a graphics card busy for 100 ms or more, and each one checked the result against a reference computed on the CPU **in a single thread**.
2. Once the first was solved, **compilation came to dominate**: 6 to 8 s per sample. A profile of `cpusim-devc` (the project's kernel compiler) showed 16% of the time in a single LLVM function, `getShuffleDemandedElts`.
3. **fdtd3d** (a 3D finite-difference benchmark) in the generic module **hung the compilation**: more than 10 minutes, consuming 4 GB of memory, without finishing.
4. After it was unblocked, the same generic fdtd3d still took **24 s** to compile.

## The idea

Each problem has its own solution; in all of them, the starting point was a profile (of the test or of `cpusim-devc`) that located the cost, followed by the removal of redundant work.

- **Tests:** a test needs to exercise the same code paths, not the same volume of data. A 1000x1000 matrix goes through the same border `if`s as a 4000x4000 one, and the CPU reference can be parallel.
- **Joining chunks:** joining 16 chunks of 16 lanes into a 256-lane vector can be done as a binary tree reduction (pairs, then pairs of pairs, log2 16 = 4 levels), instead of inserting one chunk at a time into a full-width accumulator.
- **LLVM bug:** rewrite the IR pattern that triggers the infinite loop in the backend into a semantically identical form that the backend handles without problems.
- **Rare path:** for code that almost never executes, the criterion is IR size (which determines compile time), not performance.

## How it works, step by step

### (a) Tests with small workloads and parallel references

Each sample defines its sizes with a `SAMPLE_SIZE(full, test)` macro. Compiled normally, it uses the full size; compiled with `-DSAMPLE_TEST` (which `ctest`, the CMake test runner, does), it uses the test size. Mandelbrot example:

```text
normal use:  4096 x 3072 pixels
test:        1024 x  770 pixels   (16 times less work)
```

The test sizes are deliberately not "round" (1030, 1010, 770...): numbers that are not multiples of the block size ensure that the border blocks, with leftover threads, continue to be tested.

In addition, the CPU reference now uses OpenMP, and each test gained a 120 s limit, so that a hang shows up as a failure instead of blocking the suite forever.

### (b) Joining the chunks as a tree

Many operations are generated in chunks of 16 lanes (the size of a zmm register with floats) and then joined into a vector of `W` lanes. The old version did, for each chunk, a `shufflevector` of width `W` (the LLVM IR instruction that builds a vector from a constant mask of indices over the concatenation of two operands): operand 0 was the accumulator and operand 1 the extended chunk, and the mask copied the accumulator and inserted the chunk at its position.

With `W = 256` (16 chunks), this gives 16 shuffles of 256 elements. `getShuffleDemandedElts` translates a set of demanded elements in the output of a shuffle into the sets demanded in each operand, walking the entire mask (O(W) cost per call). It is called by `computeKnownBits`, `ComputeNumSignBits`, `SimplifyDemandedVectorElts` and the like, which recurse through the operands; in a chain of 16 shuffles of width `W`, each query traverses the chain and pays O(W) per link. The total work grows with chunks x width, that is, with W²/16, and is repeated in every pass that makes these queries.

The new version joins as a tree:

```text
level 0:  p0  p1  p2  p3  p4  p5  p6  p7  ...  p15      (16 chunks of 16)
level 1:  [p0 p1] [p2 p3] [p4 p5] [p6 p7] ...            (8 vectors of 32)
level 2:  [p0 p1 p2 p3] [p4 p5 p6 p7] ...                (4 vectors of 64)
level 3:  [p0 ... p7] [p8 ... p15]                       (2 vectors of 128)
level 4:  [p0 ... p15]                                   (1 vector of 256)
```

Each level has, in total, 256 output elements, and there are only 4 levels. Instead of 16 x 256 = 4096 mask elements, there are 4 x 256 = 1024; in general, O(W log(W/16)) instead of O(W²/16). In addition, the depth of the shuffle chain drops from 16 to 4, which bounds the recursion of the analyses.

### (c) The LLVM 19 bug with wide masks

The execution mask of a variant of `W` lanes is a `<W x i1>` ([00-basic-concepts.md](../00-basic-concepts.md)). With 128 lanes or more, after O3 this pattern appeared:

```text
%m = select i1 %c, <128 x i1> %a, <128 x i1> %b     ; chooses mask a or b
%x = bitcast <128 x i1> %m to i128                   ; reads the mask as an integer
```

LLVM's instruction selection builds, per basic block, a SelectionDAG and alternates type and operation legalization phases with the DAGCombiner, which applies rewrite rules from a worklist until a fixed point; every node created or changed goes back to the worklist. On x86 with AVX-512, `<128 x i1>` is not a legal type (the largest mask register, `k`, has 64 bits with AVX512BW), so the legalizer splits the vector into `<64 x i1>` halves, and the `i128` of the bitcast is split into two `i64`. In LLVM 19, this combination of an illegal mask `select` followed by a `bitcast` makes the DAGCombiner alternately apply rewrites that undo each other on the `select`/`bitcast` pair (we did not identify the exact pair of rules; the 18-line reproducer below is enough to trigger it), and the worklist never empties; since each round allocates new nodes, memory grows without bound. The generic fdtd3d spent more than 10 minutes and 4 GB this way.

The case was isolated with `llvm-reduce` (delta debugging reduction over the IR: it removes instructions, blocks, arguments and operands as long as a test script keeps reproducing the hang) down to an 18-line file. The workaround: choose between the equivalent **integers** instead of between the masks.

```text
before:  %m = select %c, <128 x i1> %a, <128 x i1> %b
after:   %ia = bitcast %a to i128
         %ib = bitcast %b to i128
         %ir = select %c, i128 %ia, i128 %ib
         %m  = bitcast %ir to <128 x i1>
```

The semantics are identical bit for bit; only the type of the `select` changes. In the DAG, an `i128` `select` with a scalar condition is legalized as two `i64` `select`s (or `cmov`), a path that does not go through the problematic rules. The rewrite runs **after** O3 because InstCombine canonicalizes `select c, (bitcast a), (bitcast b)` back into `bitcast (select c, a, b)` when the operands have the same source type, undoing the workaround.

### (d) One gather per chunk on the rare path

When the compiler cannot prove the pattern of an access ([12-static-access-analysis.md](12-static-access-analysis.md)), it tests at run time ([06-dynamic-coalescing.md](../iteration-1/06-dynamic-coalescing.md)). In the "mixed" case, each chunk of 16 lanes tested on its own among three paths (contiguous, broadcast, gather). With 32 lanes that is 2 chunks, 6 paths; with 256 lanes it is 16 chunks, 48 paths for **each** memory access, and LLVM has to optimize all of that. For more than 2 chunks, the mixed path now directly emits one masked gather (`llvm.masked.gather`, which becomes `vgatherdps`/`vpgatherdd`) per chunk, without tests.

## The code

### The sample size

`samples/common/common.cuh:19-26`

```cpp
// Problem sizes: GPU-sized by default (>= 100 ms on an RTX 5060 where the
// kernel allows it), smaller with -DSAMPLE_TEST, which the ctest runs use:
// the same code paths and edge cases in seconds instead of minutes.
#ifdef SAMPLE_TEST
#define SAMPLE_SIZE(full, test) (test)
#else
#define SAMPLE_SIZE(full, test) (full)
#endif
```

The macro chooses the first or the second argument depending on `SAMPLE_TEST`. Each sample uses it like this (`samples/mandelbrot.cu:32`):

```cpp
    const int w = SAMPLE_SIZE(4096, 1024), h = SAMPLE_SIZE(3072, 770);
```

### The parallel reference

`samples/mandelbrot.cu:34-37`

```cpp
    std::vector<int> ref(size_t(w) * h);
#pragma omp parallel for schedule(dynamic, 8)
    for (int py = 0; py < h; ++py)
        for (int px = 0; px < w; ++px) ref[size_t(py) * w + px] = mandel_iters(px, py, w, h);
```

`schedule(dynamic, 8)` distributes the image rows in chunks of 8 on demand, which matters in mandelbrot, where the cost per row varies greatly (rows that cross the set iterate up to the limit).

### The test configuration

`CMakeLists.txt:133-144`

```cmake
if(TARGET cpusim-devc)
    # The samples carry GPU-sized workloads (>= 100 ms on an RTX 5060 where the
    # kernel allows it); the tests build them with -DSAMPLE_TEST (smaller sizes,
    # same code paths) and OpenMP for the CPU reference: seconds each.
    foreach(sample IN ITEMS matmul_naive matmul transpose heat2d scan bitonic_sort mandelbrot)
        add_test(NAME sample_${sample}
            COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tests/cuda_sample_test.sh ${CMAKE_BINARY_DIR}/cucc
                    $<TARGET_FILE:cpusimd> ${CMAKE_CURRENT_SOURCE_DIR}/samples/${sample}.cu -fmad=false -fopenmp -DSAMPLE_TEST)
        set_tests_properties(sample_${sample} PROPERTIES PASS_REGULAR_EXPRESSION "${sample}: PASS.*${sample}: PASS"
                                                         FAIL_REGULAR_EXPRESSION "FAIL" LABELS samples TIMEOUT 120)
    endforeach()
endif()
```

Each sample becomes a test compiled with `-fopenmp -DSAMPLE_TEST` and with `TIMEOUT 120` (seconds).

### The tree join

`src/compiler/simt.cpp:874-888`

```cpp
// The W-lane vector made of 16-lane parts, joined pairwise (16+16, 32+32,
// ...): one shuffle per pair instead of one W-wide shuffle per part, which
// made the optimizer's work quadratic in W.
Value* WarpBuilder::concat_chunks(const std::vector<Value*>& parts) {
    std::vector<Value*> level = parts;
    while (level.size() > 1) {
        const int n = int(cast<FixedVectorType>(level[0]->getType())->getNumElements());
        SmallVector<int, 512> idx;
        for (int i = 0; i < 2 * n; ++i) idx.push_back(i);
        std::vector<Value*> next;
        for (size_t i = 0; i + 1 < level.size(); i += 2) next.push_back(b_.CreateShuffleVector(level[i], level[i + 1], idx));
        level = std::move(next);
    }
    return level[0];
}
```

While there is more than one vector in the current level, it joins them two by two. The shuffle with mask `0, 1, ..., 2n-1` is the concatenation of the two operands, which the backend reduces to subvector insertions (`vinserti64x4` etc.) or simply to two adjacent registers during legalization. Each iteration of the `while` is one level of the tree. The function is used everywhere that builds a vector from chunks: loads, vector math functions, warp shuffles.

### The LLVM 19 bug workaround

`src/compiler/main.cpp:149-170`

```cpp
// Works around an LLVM 19 code generator hang (an endless DAG combine):
//     %m = select i1 %c, <128 x i1> %a, <128 x i1> %b
// followed by bitcasts of %m, as the masks of 128-lane and wider variants get
// after O3. The same select on the integer the mask is made of is fine. Runs
// after the optimizer, which would turn it back.
void widen_mask_selects(Module& m) {
    for (Function& f : m)
        for (BasicBlock& bb : f)
            for (Instruction& inst : make_early_inc_range(bb)) {
                auto* sel = dyn_cast<SelectInst>(&inst);
                auto* vt = sel ? dyn_cast<FixedVectorType>(sel->getType()) : nullptr;
                if (!vt || !vt->getElementType()->isIntegerTy(1) || vt->getNumElements() <= 64 ||
                    sel->getCondition()->getType()->isVectorTy())
                    continue;
                IRBuilder<> b(sel);
                Type* it = b.getIntNTy(vt->getNumElements());
                Value* r = b.CreateSelect(sel->getCondition(), b.CreateBitCast(sel->getTrueValue(), it),
                                          b.CreateBitCast(sel->getFalseValue(), it));
                sel->replaceAllUsesWith(b.CreateBitCast(r, vt));
                sel->eraseFromParent();
            }
}
```

It walks all the instructions of the module looking for a `select` whose result is a bit vector (`i1`) with more than 64 elements and whose condition is a single bit (not a vector). Each one is replaced by the integer version: it converts the two options to an `N`-bit integer, chooses, converts back. `make_early_inc_range` advances the iterator before handing out the element, which allows erasing the current instruction during the traversal.

And where it is called, `src/compiler/main.cpp:497-498`:

```cpp
    run_pipeline(*m, tm.get(), "O3");
    widen_mask_selects(*m);
```

Right after O3 and before code generation, as the comment says.

### The rare path with one gather per chunk

`src/compiler/simt.cpp:1389-1399`

```cpp
        std::vector<Value*> parts(chunks);
        for (unsigned h = 0; h < chunks; ++h) {
            // Wider than a warp: one gather per chunk. Testing each chunk (3
            // paths each) multiplies the code by the chunks: the generic
            // fdtd3d module took 24 s to compile instead of 7. Only the
            // module without a known block shape gets here.
            if (chunks > 2) {
                parts[h] = b_.CreateMaskedGather(v16, b_.CreateGEP(b_.getInt8Ty(), addr.base, offs[h]), align, masks[h],
                                                 PoisonValue::get(v16));
                continue;
            }
```

`chunks > 2` means "more than 32 lanes" (more than one warp). In that case each chunk goes straight to a masked gather. With 32 lanes or fewer, the three-path test per chunk remains, which is cheap at that size. The same holds for stores, with one scatter per chunk (`src/compiler/simt.cpp:1479`).

## When it applies (and when it does not)

- **(a)** It applies only to the `ctest` tests. Running a sample by hand, without `-DSAMPLE_TEST`, still uses the full workload, which is the one used in the performance measurements.
- **(b)** It applies to every chunk join. With 32 lanes (2 chunks) the tree has a single level and nothing changes; the gain grows with the width.
- **(c)** It only affects `select` of masks with more than 64 lanes and a scalar condition, that is, the variants of 128 lanes or more. If a future LLVM version fixes the bug, the function can be removed.
- **(d)** It only affects the "mixed" path of loads and stores, in variants with more than 32 lanes. As the comment says, in practice only the generic module (without a known block shape) gets there: in the JIT modules, static analysis resolves most accesses beforehand.

## Results

- **Sample test suite:** from **438 s** (mandelbrot alone at 232 s) to **32 s**. Today the suite takes about **55 s**, because more tests have been added since then.
- **Tree join:** scan compilation from **6.3 s to 4.4 s**. The function `getShuffleDemandedElts` accounted for 16% of the time of `cpusim-devc` before the change.
- **LLVM 19 bug workaround:** the generic fdtd3d went from "hung for more than 10 minutes with 4 GB" to **compiling in 6.9 s**.
- **One gather per chunk on the rare path:** generic fdtd3d from **24 s to 7 s** of compilation.

## Limitations and trade-offs

- Tests with smaller workloads may fail to catch a defect that only appears with a lot of data (for example an index that only overflows above 2^31 elements). For this reason the test sizes were chosen to keep the border cases, and the performance measurements continue to use the full workload.
- On the rare path at large widths, a chunk that happened to be contiguous is now read with a gather, which is slower than a vector load. The trade-off was accepted because this path practically only runs in the generic module, which is already the fallback option.
- The LLVM bug workaround depends on running after O3. If the pipeline order changes, the problem may come back.

## See also

- [00-basic-concepts.md](../00-basic-concepts.md): lanes, masks and registers.
- [01-specialized-jit.md](../iteration-1/01-specialized-jit.md): the JIT module and the generic module.
- [02-width-variants.md](../iteration-1/02-width-variants.md): the 64, 128, 256 and 512 lane variants, where the compilation problems appeared.
- [06-dynamic-coalescing.md](../iteration-1/06-dynamic-coalescing.md): the run-time pattern tests.
- [11-llvm-tuning.md](../iteration-1/11-llvm-tuning.md): other adaptations to LLVM's behavior.
- [12-static-access-analysis.md](12-static-access-analysis.md): why the rare path almost never runs in the JIT modules.
- [22-what-did-not-work.md](22-what-did-not-work.md): ideas that did not work.
