# 11. LLVM tuning: asking the optimizer for what it already knows how to do

> **In one sentence:** the `cpusim-devc` compiler uses LLVM 19 underneath, and a few small adjustments to how it is invoked (a larger limit for "versioning" loops, an option so that `instcombine` does not abort, an extra pipeline that unrolls short loops and breaks up struct copies in kernels with local arrays) let LLVM perform optimizations that, with the default configuration, it would skip.

The project's general concepts (lane, uniform/varying, warp variants) are in [00-basic-concepts.md](../00-basic-concepts.md).

> **Iteration:** `-unswitch-threshold` and `instcombine<no-verify-fixpoint>` are from iteration 1. The extra pipeline for local arrays (unrolling fixed loops and `expand_local_memcpys`) arrived in [iteration 2](../iteration-2/README.md).

## The problem

`cpusim-devc` uses LLVM 19's new pass manager, with pipelines described in text (the syntax of `opt -passes=...`). The default thresholds and heuristics of the passes were calibrated for typical C/C++ code compiled by `clang`. The code that `cpusim-devc` generates differs in three ways:

1. **Invariant tests inside loops.** The warp code decides, at run time, whether a memory access is contiguous, whether all lanes read the same address, etc. ([06-dynamic-coalescing.md](06-dynamic-coalescing.md)). These tests are loop-invariant, but they stay inside the loop and are reevaluated on every iteration; LICM does not resolve them, because what needs to leave the loop is the branch, not just the computation of the condition.
2. **Per-thread local arrays.** A CUDA kernel can declare `float v[4];` per thread. In the vectorized version, each lane has its own copy; if the array remains an `alloca`, each access becomes a load/store (with a varying index, one gather/scatter per lane), when the ideal would be to promote it to SSA values in registers.
3. **Struct copies.** `Body p = bodies[i];` (as in `nbody`) becomes an `llvm.memcpy` intrinsic in the IR; in our pipeline, the local variable copied this way stayed in memory after SROA.

There is a fourth point, about **correctness** rather than performance: in LLVM 19, `instcombine` invoked by a textual pipeline checks by default that it reached a fixed point (no applicable rule) within the allowed number of iterations, and **aborts the compilation** with a fatal error if it did not. In large, atypical code such as the code generated here, this happened.

## The idea

The necessary passes already exist in LLVM; the problem is that their heuristics or preconditions are not satisfied by the generated code. The adjustments are:

- **raise the cost threshold** of loop unswitching (`-unswitch-threshold=2000`), so that large loops with invariant tests get versioned;
- **disable the fixed-point verification** of `instcombine` (`instcombine<no-verify-fixpoint>`), which turns a benign non-convergence into a fatal error;
- **establish SROA's preconditions** before calling it: expand small `memcpy`s into loads and stores with constant offsets and unroll constant-trip-count loops, so that all accesses to the `alloca`s have constant indices.

## How it works, step by step

### Where each piece comes in

`cpusim-devc` runs LLVM in several stages (file `src/compiler/main.cpp`):

```text
1. read LLVM options (-unswitch-threshold=2000 + CPUSIM_LLVM_FLAGS)
2. per-thread code (as CUDA wrote it):
     always-inline, globaldce
     cleanup: sroa, early-cse, instcombine<no-verify-fixpoint>, simplifycfg, licm, gvn, ...
3. only in kernels that still have an alloca (local arrays/structs):
     expand_local_memcpys
     loop(loop-rotate, indvars, loop-unroll-full), sroa, ... cleanup
4. SIMT transformation: generates the warp variants (simt.cpp)
5. full LLVM O3 over the warp code  <- unswitch acts here
6. machine code generation (AVX-512)
```

### `-unswitch-threshold=2000`: taking tests out of loops

*Loop unswitching* (`SimpleLoopUnswitch` in LLVM) versions a loop on a condition that is invariant in it: the branch is hoisted to before the loop and the loop is cloned, one copy per branch outcome. Example:

```cpp
for (int k = 0; k < 16; ++k) {
    if (contiguous) acc += load_vector(p + k);   // 1 instruction
    else            acc += load_lane_by_lane(p + k);  // 16 instructions
}
```

`contiguous` is decided before the loop and does not change. Unswitching turns this into:

```cpp
if (contiguous) for (int k = 0; k < 16; ++k) acc += load_vector(p + k);
else            for (int k = 0; k < 16; ++k) acc += load_lane_by_lane(p + k);
```

Now the test is done once, and each version of the loop is free of the branch, which also enables other optimizations in the body (vectorization, scheduling). The price is code duplication. LLVM only applies non-trivial unswitching if the estimated cost of the loop (the sum of the instruction costs according to the `TargetTransformInfo`) is below the threshold; the default is low for warp code, whose loops are large and contain several of these tests. `cpusim-devc` passes `-unswitch-threshold=2000` to LLVM, raising the limit.

### `CPUSIM_LLVM_FLAGS`: experimenting without recompiling

The environment variable `CPUSIM_LLVM_FLAGS` allows passing other LLVM internal options, separated by spaces, without touching the code. For example, `CPUSIM_LLVM_FLAGS="-unswitch-threshold=200"` goes back to a lower limit, for comparison. Since the extra options come after the default, they override it.

### `instcombine<no-verify-fixpoint>`: do not abort

`instcombine` is LLVM's peephole simplifier over the IR (for example `x * 2` becomes `x << 1`, `(a + 1) - 1` becomes `a`). It uses a worklist and reapplies rules until a fixed point, but a cycle of rules or a simplification that is only discovered on a new visit can prevent convergence within the iteration limit. In LLVM 19, the fixed-point verification is on by default on this path and, if the limit is reached, it raises a **fatal error**. The parameter `<no-verify-fixpoint>` turns this verification off: `instcombine` does what it can and the pipeline goes on. It appears in both cleanup stages of the per-thread code (steps 2 and 3 above).

### Extra pipeline for kernels with local arrays

Consider a thread with a 4-element array:

```cpp
float v[4];
for (int i = 0; i < 4; ++i) v[i] = a[base + i];
for (int i = 0; i < 4; ++i) sum += v[i] * v[i];
```

`SROA` (*Scalar Replacement of Aggregates*) partitions an `alloca` into slices by the offsets actually accessed and promotes each slice to SSA values (with phis inserted as in `mem2reg`). For this, every access to the `alloca` must have a **constant** offset; a GEP with a variable index (`v[i]`) makes SROA give up on the whole `alloca`. If the loop is unrolled first:

```cpp
v[0] = a[base + 0]; v[1] = a[base + 1]; v[2] = a[base + 2]; v[3] = a[base + 3];
sum += v[0]*v[0]; sum += v[1]*v[1]; sum += v[2]*v[2]; sum += v[3]*v[3];
```

the indices become constants and SROA eliminates the array entirely. For this reason the extra pipeline is:

```text
loop-simplify, lcssa,                      (leaves the loops in canonical form)
loop(loop-rotate, indvars, loop-unroll-full),  (discovers the trip count and unrolls)
sroa<modify-cfg>,                          (array -> registers)
early-cse, instcombine<no-verify-fixpoint>, simplifycfg, gvn, ... (cleanup)
```

`loop-simplify` and `lcssa` put the loops in the canonical form (single preheader, dedicated exits, LCSSA) required by the loop passes; `loop-rotate` converts the loop to the do-while form with the test at the end; `indvars` canonicalizes the induction variable and, with SCEV, computes the iteration count; `loop-unroll-full` unrolls completely when that count is a small constant.

This pipeline runs **only in kernels that still have an `alloca`** after the initial cleanup. In the others, the loops stay as in the source: unrolling unnecessarily would get in the way of the SIMT transformation, which handles whole loops (for example, as [uniform loops](07-uniform-loops.md)) and whose linearization cost grows with the number of blocks.

### `expand_local_memcpys`: expanding struct copies

In `nbody`, something like

```cpp
Body p = bodies[i];   // struct with x, y, z, mass: 16 bytes
```

becomes `memcpy(&p, &bodies[i], 16)` in the IR. In our pipeline, `p` remained an `alloca` after SROA (observed in the generated IR; we did not isolate which internal SROA condition prevents promotion in this case). The function `expand_local_memcpys` replaces the `memcpy` with load/store pairs of 4 bytes:

```text
before: memcpy(&p, &bodies[i], 16)
after:  p.bytes[0..3]   = bodies[i].bytes[0..3]
        p.bytes[4..7]   = bodies[i].bytes[4..7]
        p.bytes[8..11]  = bodies[i].bytes[8..11]
        p.bytes[12..15] = bodies[i].bytes[12..15]
```

Now SROA sees four `i32` accesses at constant offsets and promotes `p` to four SSA values (the `bitcast`s to `float` are resolved by `instcombine`). It only acts on small copies (up to 256 bytes), of known size and a multiple of 4, non-volatile, and in which the source or the destination is a local variable.

## The code

### The LLVM options

`src/compiler/main.cpp:305-324`

```cpp
    // Tuning of LLVM's own passes for warp code. CPUSIM_LLVM_FLAGS adds more
    // (for experiments), e.g. "-unswitch-threshold=200".
    {
        // Loops in warp code test invariant access-pattern flags; letting LLVM
        // version them (unswitch) takes those tests out of the loops.
        std::vector<std::string> flags = {"cpusim-devc", "-unswitch-threshold=2000"};
        if (const char* extra = std::getenv("CPUSIM_LLVM_FLAGS")) {
            std::string all = extra;
            size_t pos = 0;
            while (pos < all.size()) {
                size_t end = all.find(' ', pos);
                if (end == std::string::npos) end = all.size();
                if (end > pos) flags.push_back(all.substr(pos, end - pos));
                pos = end + 1;
            }
        }
        std::vector<const char*> argv2;
        for (auto& f : flags) argv2.push_back(f.c_str());
        cl::ParseCommandLineOptions(int(argv2.size()), argv2.data());
    }
```

LLVM's internal options are registered via `cl::opt` and read as command-line arguments. The code builds a synthetic `argv`: the program name, the unswitch limit and, if present, each word of `CPUSIM_LLVM_FLAGS`. It then hands everything to `cl::ParseCommandLineOptions`. This happens once, before any pass runs; since the options are global to the process, they apply to all subsequent pipelines.

### Cleanup of the per-thread code

`src/compiler/main.cpp:352-361`

```cpp
    // Per-thread clean up before vectorizing: inline everything, promote locals
    // to registers, simplify. No unrolling or vectorization here: those would
    // obscure the per-thread structure the SIMT pass works on.
    run_pipeline(*m, tm.get(), "always-inline,globaldce");
    for (auto& k : kernels) cpusim::lower_shared_memory(*m, k);
    run_pipeline(*m, tm.get(),
                 "globaldce,"
                 "function(sroa<modify-cfg>,early-cse<memssa>,instcombine<no-verify-fixpoint>,simplifycfg,"
                 "loop-mssa(licm<allowspeculation>),gvn,instcombine<no-verify-fixpoint>,simplifycfg,dse,adce,"
                 "lower-switch,lcssa)");
```

The pipeline is described in text, in the new pass manager's syntax. Both `instcombine<no-verify-fixpoint>` appear here. The comment explains why there is no unrolling at this stage: it applies to all kernels, and the SIMT transformation needs the original structure of the loops.

### The extra pipeline, only for kernels with local arrays

`src/compiler/main.cpp:362-373`

```cpp
    // Per-thread arrays (float v[4]) usually live through loops of a fixed
    // trip count: unrolled, their indices are constants and SROA turns them
    // into registers. Only kernels that still have arrays, so the others keep
    // their loops as written.
    for (auto& k : kernels)
        if (has_alloca(*k.scalar)) {
            expand_local_memcpys(*k.scalar);
            run_function_pipeline(*k.scalar, tm.get(),
                                  "loop-simplify,lcssa,loop(loop-rotate,indvars,loop-unroll-full),sroa<modify-cfg>,"
                                  "early-cse<memssa>,instcombine<no-verify-fixpoint>,simplifycfg,gvn,"
                                  "instcombine<no-verify-fixpoint>,simplifycfg,dse,adce,lower-switch,lcssa");
        }
```

For each kernel, if any `alloca` is left (`has_alloca`), the small copies are expanded first, and then the pipeline runs that unrolls fixed-size loops (`loop-unroll-full`) **before** `sroa`. The order is the central point: unrolling after SROA would be useless, because SROA would already have given up.

`src/compiler/main.cpp:200-205`

```cpp
bool has_alloca(const Function& f) {
    for (const BasicBlock& bb : f)
        for (const Instruction& i : bb)
            if (isa<AllocaInst>(i)) return true;
    return false;
}
```

A simple test: is there any `alloca` instruction in the kernel? The previous cleanup (which already includes an SROA) promoted the `alloca`s with constant accesses; those that remain are candidates for the extra treatment.

### Expanding `memcpy`

`src/compiler/main.cpp:172-198`

```cpp
// Small fixed-size copies to or from a local (struct copies: `auto p = a[i]`,
// `a[i] = p`) as 4-byte loads and stores, which SROA then turns into
// registers; it leaves the local in memory when it is copied with memcpy.
bool expand_local_memcpys(Function& f) {
    bool changed = false;
    for (BasicBlock& bb : f)
        for (Instruction& inst : make_early_inc_range(bb)) {
            auto* cpy = dyn_cast<MemCpyInst>(&inst);
            auto* len = cpy ? dyn_cast<ConstantInt>(cpy->getLength()) : nullptr;
            if (!len || len->getZExtValue() > 256 || len->getZExtValue() % 4 != 0 || cpy->isVolatile()) continue;
            if (!isa<AllocaInst>(getUnderlyingObject(cpy->getDest())) &&
                !isa<AllocaInst>(getUnderlyingObject(cpy->getSource())))
                continue;
            IRBuilder<> b(cpy);
            Type* i32 = b.getInt32Ty();
            const Align da = cpy->getDestAlign().valueOrOne(), sa = cpy->getSourceAlign().valueOrOne();
            for (uint64_t off = 0; off < len->getZExtValue(); off += 4) {
                Value* v = b.CreateAlignedLoad(i32, b.CreateConstInBoundsGEP1_64(b.getInt8Ty(), cpy->getSource(), off),
                                               commonAlignment(sa, off));
                b.CreateAlignedStore(v, b.CreateConstInBoundsGEP1_64(b.getInt8Ty(), cpy->getDest(), off),
                                     commonAlignment(da, off));
            }
            cpy->eraseFromParent();
            changed = true;
        }
    return changed;
}
```

For each `memcpy`: the size must be a constant, at most 256 bytes, a multiple of 4, and the copy must not be volatile. The source or the destination must have an `AllocaInst` as its underlying object (`getUnderlyingObject`). Passing these tests, the `memcpy` becomes a sequence of `i32` load/store pairs with the alignment of each offset computed by `commonAlignment`, and the original `memcpy` is deleted.

## When it applies (and when it does not)

- **`-unswitch-threshold=2000`** applies to all kernels, but only makes a difference where there is a loop with an invariant test that LLVM decides to version. It acts in the final `O3`, over the warp code already generated.
- **`instcombine<no-verify-fixpoint>`** is only a safeguard. It does not change the generated code when `instcombine` converges; it avoids the abort when it does not.
- **The extra pipeline** only runs in kernels that still have an `alloca` after the cleanup. Loops without a constant trip count are not unrolled by `loop-unroll-full`, and the corresponding array stays in memory (the compiler then keeps one copy per lane; see [09-state-at-barriers.md](09-state-at-barriers.md) for the case with barriers, in which that copy goes to the frame).
- **`expand_local_memcpys`** does not act on copies larger than 256 bytes, of variable size, of size not a multiple of 4, volatile, or between two regions that are not `alloca`s (for example, global to global).
- **`CPUSIM_LLVM_FLAGS`** is for experiments; invalid options make LLVM reject the command line.

## Results

There is no isolated measurement of these adjustments recorded. Qualitatively:

- Unswitching with a higher limit takes the access pattern tests out of the loops, where otherwise they would be redone on every trip.
- The extra pipeline and `expand_local_memcpys` let fixed-size local arrays and struct copies (like the one in `nbody`) be promoted to registers instead of staying in memory.
- `instcombine<no-verify-fixpoint>` is a matter of correctness: without it, LLVM 19 aborted the compilation of some kernels.

## Limitations and trade-offs

- **Larger code.** Versioning loops duplicates their body; unrolling loops also increases the code. This increases compile time ([21-compile-time.md](../iteration-2/21-compile-time.md)) and may increase pressure on the L1I and the micro-op cache.
- **Fixed limit.** The value 2000 is a chosen number, not computed per kernel. `CPUSIM_LLVM_FLAGS` exists precisely to test other values.
- **Selectivity by `alloca`.** Unrolling only applies to kernels with remaining local variables. A kernel without arrays, but with a short loop that would benefit from unrolling, is not affected by this stage (the final `O3` may still unroll it on its own).

## See also

- [00-basic-concepts.md](../00-basic-concepts.md)
- [01-specialized-jit.md](01-specialized-jit.md): the compiler runs again at launch, with the block shape known.
- [06-dynamic-coalescing.md](06-dynamic-coalescing.md): the access pattern tests that unswitching takes out of the loops.
- [07-uniform-loops.md](07-uniform-loops.md): why the structure of the loops matters for the SIMT transformation.
- [09-state-at-barriers.md](09-state-at-barriers.md): remaining local arrays go to the frame.
- [21-compile-time.md](../iteration-2/21-compile-time.md): the cost of compiling more code.
