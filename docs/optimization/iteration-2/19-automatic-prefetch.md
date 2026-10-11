# 19. Automatic prefetch

> **In one sentence:** the compiler now looks, inside loops, for memory accesses that advance by a fixed step on each iteration and that are spread across the lanes (each lane on a different cache line); for these, it inserts "fetch this ahead of time" instructions (prefetch) a few dozen iterations ahead, one per lane, because the hardware prefetcher cannot keep up with so many streams at once.

The project's general concepts (lane, coalesced access, gather/scatter) are in [00-basic-concepts.md](../00-basic-concepts.md). The specialized concepts of this document:

- **Zen 4 hardware prefetchers**: the L1D has a stream prefetcher (next line on ascending accesses), a stride prefetcher indexed by the instruction address (IP) and a region prefetcher; the L2 has a stream prefetcher and an "up/down" one (adjacent line). The stride prefetcher keeps, per load IP, the last address and the last delta, and only issues prefetches after seeing the same delta repeated; the stream ones track a limited number of sequential streams (dozens, not hundreds) and do not cross 4 KiB page boundaries.
- **Why they do not follow gathers**: a `vgatherdps` is a single instruction (one IP) that generates 16 addresses per execution. For the stride prefetcher, the sequence of addresses seen at this IP has lane-to-lane deltas (+20000 bytes, for example) interleaved with the jump back to lane 0 of the next iteration, so there is no stable delta to learn. For the stream prefetchers, each lane is an independent stream; with 128 simultaneous streams per worker (see below) the table is insufficient, and when the step per iteration is larger than 4 KiB (going down a column of 5000 floats) each access lands on a new page, which they do not follow.
- **Software prefetch**: `llvm.prefetch(addr, rw, locality, cachetype)` becomes `prefetcht0/t1/t2/nta` (or `prefetchw` for writes) on x86. It is a hint with no architectural effect: it does not fault, does not change visible state, and is dropped if the address has no translation in the TLB that can be resolved without a fault.
- **SCEV (ScalarEvolution)**: an LLVM analysis that represents integer values (and addresses) in a loop as closed-form expressions. The central form is the **AddRec** `{start,+,step}<L>`: the value is `start` on the first iteration of loop `L` and adds `step` on each iteration; it is **affine** when `step` is invariant in `L` (otherwise it is a higher-order recurrence, such as `{0,+,{1,+,1}}`). `start` and `step` can be symbolic (`n * 4`), and values that the analysis does not decompose appear as `SCEVUnknown` leaves. For example, the address of `m[i * n]` inside a loop over `i` is `{m,+,(4 * n)}<L>`.

## The problem

This section tells the investigation in the order in which it happened, because the path teaches as much as the result.

### A suspicious tie

After splitting blocks into warps ([16-splitting-blocks-into-warps.md](16-splitting-blocks-into-warps.md)), `matrix-rotate` (rotation of a 5000 x 5000 matrix by layers) already ran in about 5.3 to 5.5 ms. The OpenMP version of the same program, on the same machine, took 4.1 to 4.2 ms.

First suspicion: OpenMP uses 32 CPUs, we use 28. Running OpenMP with the same 28 CPUs, it took **5.1 to 5.6 ms**: a tie. So was the difference only the CPUs?

Second check: if the limit were the number of CPUs, our time should change a lot with the number of cores. Comparing with a run restricted to **16 cores**, giving more cores to our side **did not improve** the time. That is, it was not a lack of cores: something was stalling each core.

Third check: perhaps some workers were sitting idle (imbalance). With `CPUSIM_VERBOSE=2`, the simulator prints how long the workers spent busy: **90 to 94%**. It was not imbalance.

### The hypothesis: the prefetcher cannot keep up

Looking at how each side accesses memory:

- An **OpenMP thread** processes one layer at a time. It traverses **4 streams** (the four sides of the layer), each with a fixed step. The hardware prefetcher recognizes these 4 patterns easily and fetches the data beforehand.
- One **warp of ours** processes 32 layers **at the same time** (one per lane). Each read is a **gather** of 16 addresses (an AVX-512 register has 16 lanes), and each of these 16 addresses is on a different cache line, because each lane is on a different layer.

```text
OpenMP (1 thread):          our warp (16 lanes of a register):
  top side     -> stream 1      lane 0: layer 0, line X0
  right side   -> stream 2      lane 1: layer 1, line X1   (another line)
  bottom side  -> stream 3      lane 2: layer 2, line X2   (another line)
  left side    -> stream 4      ...  16 lines per instruction
                                x 4 sides x 2 halves = 128 streams per worker
```

The stride prefetcher learns **per instruction** (per IP), and the gather's IP produces 16 scattered addresses per execution, with no stable delta; the stream ones cannot hold 128 streams nor follow steps larger than a page. Result: each line is only fetched at the moment of the gather, and each gather waits for the DRAM latency of the missing lines.

### The manual experiment

To test the hypothesis, we put `__builtin_prefetch` by hand in a copy of the kernel, a certain number of iterations ahead. First result: **-15%** in time, but with only **1 prefetch per chunk** of 16 lanes.

Investigating why only 1, we discovered that the compiler turned a 16-lane prefetch into a single prefetch, at the address of the **first lane**. This makes sense for coalesced accesses (16 neighboring lanes fit in one or two cache lines), but here it left 15 of the 16 lines out.

We changed the compiler to generate **1 prefetch per lane** when the lanes are spread out. With that, and with a distance of 32 iterations, the manual prefetch gave a **median of 4.55 ms against OpenMP's 5.03 ms** (5 runs alternating between the two).

The hypothesis was right. What remained was to do this automatically, without touching the user's program.

### LLVM did not help

LLVM has a pass called `loop-data-prefetch` that inserts prefetches in loops. We enabled it and it **inserted nothing**. This pass only accepts AddRecs whose `step` is a `SCEVConstant`, that is, a step known at compile time. In `matrix-rotate`, going down a column of the matrix advances `n * 4` bytes per iteration, where `n` is only known when the program runs.

So we wrote our own.

## The idea

For each innermost loop of the kernel, look for memory accesses that:

1. go to **global memory** (not shared memory, not the thread's local arrays, which already stay in the L1 cache);
2. advance by a **fixed step per iteration** of the loop (even if this step is only known at run time, such as `n * 4`);
3. are **spread across the lanes**: from one lane to the neighbor, the address changes by 64 bytes or more (a whole cache line), or changes by an amount proportional to a size that is only known at run time.

For these, insert a prefetch of the address "K iterations from now", with K chosen to be about 1000 instructions ahead.

Condition 3 is the **check** that keeps the prefetch from getting in the way. If the lanes read neighboring addresses (coalesced access), the load becomes a contiguous read of one or two lines, which the stream prefetchers follow, and an extra prefetch would only waste issue slots. If all lanes read the same address (broadcast), it is not needed either. If it cannot be known, it does not take the risk.

## How it works, step by step

Reduced example: a warp of 4 lanes going down 4 columns of a 4x4 matrix of `float` (4 bytes), with `n = 4`:

```c
for (int i = 0; i < n; ++i)
    soma += m[i * n + threadIdx.x * n];   // each lane goes down a "lying down" column
```

Simplifying: the address is `m + 4 * (i*n + tid*n)`.

**Step 1: how much does the address move per iteration?** (`byte_step`)

Differentiating with respect to `i`: `4 * n` bytes. With `n = 4`, 16 bytes. It is not constant at compile time (it depends on `n`), but it is **fixed within the loop**. It qualifies.

**Step 2: how much does the address change from one lane to the neighbor?** (`lane_step`)

Differentiating with respect to `threadIdx.x`: `4 * n` bytes. It is proportional to `n`, a size known only at run time. By the rule, it qualifies (in a real matrix, `n = 5000` would give 20000 bytes between lanes: each lane on a different line).

Compare with a coalesced access, `m[i * n + threadIdx.x]`: the `lane_step` would be 4 bytes (a constant smaller than 64), and the prefetch would **not** be inserted.

**Step 3: how far ahead?**

If the loop body has 40 instructions: `1000 / 40 = 25` iterations ahead. The value is clamped between 8 and 64.

**Step 4: insert.** Before the access, the compiler generates `prefetch(address + 25 * step)` and marks the instruction with the tag `cpusim.auto`.

**Step 5: vectorize.** When the kernel is transformed into AVX-512 code, a prefetch with the tag `cpusim.auto` becomes **one prefetch per lane** (16 per register), each only if that lane is active.

## The code

### The overview

`src/compiler/simt.cpp:289-303`

```cpp
// Candidate software prefetches for global memory streams in loops: an access
// in an innermost loop whose per-thread address moves by a fixed step each
// iteration (constant or uniform, e.g. n * 4 down a column) gets a prefetch of
// the address a few iterations ahead. They are marked "cpusim.auto" and kept
// at vectorization only where the lanes' addresses are spread out (one line
// per lane: hardware prefetchers, which learn per instruction, do not follow
// 16 streams of one gather); for lanes next to each other they are dropped,
// the hardware already handles those. matrix-rotate: 5.45 -> 4.55 ms.
bool WarpBuilder::insert_prefetches() {
    bool inserted = false;
    TargetLibraryInfoImpl tlii(Triple(m_.getTargetTriple()));
    TargetLibraryInfo tli(tlii);
    AssumptionCache ac(f_);
    ScalarEvolution se(f_, tli, ac, *dt_, *li_);
    SCEVExpander expander(se, m_.getDataLayout(), "pf");
```

The comment summarizes the whole idea. The number "5.45 -> 4.55 ms" is from the manual experiment with distance 32; the result of the automatic one is in the "Results" section. The last lines create the `ScalarEvolution` analysis and a `SCEVExpander`, which turns a SCEV formula back into instructions.

### `lane_step`: the distance between neighboring lanes

`src/compiler/simt.cpp:306-321` (the rest of the function, up to line 346, handles sums, products and type conversions in the same way)

```cpp
    // Bytes the address moves from one lane to the next (threadIdx.x + 1),
    // through sums, constant factors, extensions and recurrences (the
    // start's). Unknown values that are uniform do not move; anything else
    // varying makes it unknown (null).
    std::function<const SCEV*(const SCEV*)> lane_step = [&](const SCEV* x) -> const SCEV* {
        if (auto* u = dyn_cast<SCEVUnknown>(x)) {
            if (special_reg(u->getValue()) == SpecialReg::TidX) return se.getOne(i64);
            return uni_->is_varying(u->getValue()) ? nullptr : se.getZero(i64);
        }
        if (isa<SCEVConstant>(x)) return se.getZero(i64);
        if (auto* r = dyn_cast<SCEVAddRecExpr>(x)) {
            if (!r->isAffine()) return nullptr;
            const SCEV* st = lane_step(r->getStepRecurrence(se));
            if (!st || !st->isZero()) return nullptr;  // the step itself varies across lanes
            return lane_step(r->getStart());
        }
```

It is a "derivative" with respect to `threadIdx.x`, taken over the address formula:

- `threadIdx.x` is worth 1 (it itself changes by 1 from one lane to the next).
- A constant, or a value that the uniformity analysis ([07-uniform-loops.md](../iteration-1/07-uniform-loops.md)) guarantees to be equal in all lanes (such as `n`), is worth 0.
- Any other value that varies between lanes in an unknown way returns "unknown" (`nullptr`).
- For a loop recurrence (`start + step * i`), the step cannot vary between lanes; if it does not vary, the answer is the derivative of the start.

On lines 325-340, the product follows the product rule with at most one moving factor: `4 * n * tid` gives `4 * n`.

### `byte_step`: how much the address moves per iteration

`src/compiler/simt.cpp:353-381`

```cpp
    // Bytes the address moves per iteration of `l`, as an i64 SCEV: through
    // sums, constant factors and extensions (CUDA indexes are int: the
    // recurrence sits inside a sext). Ignoring wrap-around is fine for a
    // prefetch, a hint that cannot fault. Null if unknown.
    std::function<const SCEV*(const SCEV*)> byte_step = [&](const SCEV* x) -> const SCEV* {
        if (se.isLoopInvariant(x, cur)) return se.getZero(i64);
        if (auto* r = dyn_cast<SCEVAddRecExpr>(x)) {
            if (r->getLoop() != cur || !r->isAffine()) return nullptr;
            return se.getTruncateOrSignExtend(r->getStepRecurrence(se), i64);
        }
        if (auto* a = dyn_cast<SCEVAddExpr>(x)) {
            // ... sum of the steps of each term
        }
        if (auto* m = dyn_cast<SCEVMulExpr>(x); m && m->getNumOperands() == 2 && isa<SCEVConstant>(m->getOperand(0))) {
            const SCEV* st = byte_step(m->getOperand(1));
            return st ? se.getMulExpr(se.getTruncateOrSignExtend(m->getOperand(0), i64), st) : nullptr;
        }
        if (auto* c = dyn_cast<SCEVCastExpr>(x)) {
            const SCEV* st = byte_step(c->getOperand(0));
            return st ? se.getTruncateOrSignExtend(st, i64) : nullptr;
        }
        return nullptr;
    };
```

Why not use directly what SCEV gives? In CUDA the indices are `int` (32 bits) and the address is 64 bits, so the index appears inside a `sext` before the GEP. SCEV only rewrites `sext({a,+,b})` as `{sext(a),+,sext(b)}` when it can prove that the recurrence does not overflow in 32 bits (`nsw` flag on the AddRec); without that proof, the expression stays as `(sext i32 {a,+,b} to i64)`, which is not a 64-bit AddRec, and `getStepRecurrence` does not apply to the address. `byte_step` goes through sums, multiplications by constants and casts, and assumes there is no wrap-around. As the comment says, for a prefetch this is acceptable: the prefetch has no architectural effect, and a wrong address costs at most one useless line brought into the cache.

### The choice and the insertion

`src/compiler/simt.cpp:383-389`

```cpp
    for (Loop* l : li_->getLoopsInPreorder()) {
        if (!l->isInnermost()) continue;
        cur = l;
        uint64_t insts = 0;
        for (BasicBlock* bb : l->blocks()) insts += bb->size();
        // About a thousand instructions ahead, 8 to 64 iterations.
        const int64_t ahead = std::clamp<int64_t>(int64_t(1000 / std::max<uint64_t>(insts, 1)), 8, 64);
```

Only innermost loops. The distance `ahead` is computed from the size of the body.

`src/compiler/simt.cpp:396-423`

```cpp
                if (auto* ld = dyn_cast<LoadInst>(&inst)) ptr = ld->getPointerOperand();
                else if (auto* st = dyn_cast<StoreInst>(&inst)) ptr = st->getPointerOperand(), write = true;
                if (!ptr || ptr->getType()->getPointerAddressSpace() != 0) continue;
                // Not shared memory or per-thread arrays: those stay in L1.
                const Value* obj = getUnderlyingObject(ptr, 12);
                if (isa<AllocaInst>(obj)) continue;
                if (auto* c = dyn_cast<CallInst>(obj); c && c->getCalledFunction() &&
                                                        c->getCalledFunction()->getName() == kSharedBaseFn)
                    continue;
                const SCEV* step = byte_step(se.getSCEV(ptr));
                if (!step || step->isZero() || !se.isLoopInvariant(step, l)) continue;
                const SCEV* offset = se.getMulExpr(se.getConstant(step->getType(), ahead), step);
                // Only where neighbouring lanes are a cache line or more
                // apart (or a run-time size apart, as down a column): there
                // the hardware prefetchers do not follow. Adjacent or equal
                // addresses across lanes (coalesced, broadcast): left to them.
                const SCEV* across = lane_step(se.getSCEV(ptr));
                if (!across || across->isZero()) continue;
                if (auto* c = dyn_cast<SCEVConstant>(across); c && c->getAPInt().abs().ult(64)) continue;
                if (!done.insert(se.getAddExpr(se.getSCEV(ptr), offset)).second) continue;
                if (!expander.isSafeToExpandAt(offset, &inst)) continue;
                IRBuilder<> eb(&inst);
                Value* at = eb.CreateGEP(eb.getInt8Ty(), ptr,
                                         expander.expandCodeFor(offset, eb.getInt64Ty(), inst.getIterator()), "pf.addr");
                if (!pf) pf = Intrinsic::getDeclaration(&m_, Intrinsic::prefetch, {ptr->getType()});
                IRBuilder<> b(&inst);
                CallInst* call = b.CreateCall(pf, {at, b.getInt32(write ? 1 : 0), b.getInt32(3), b.getInt32(1)});
                call->setMetadata("cpusim.auto", MDNode::get(ctx_, {}));
```

Reading the checks in order, an access only gets a prefetch if:

| Check | Line | If it fails |
|---|---|---|
| it is a read or write to ordinary memory (address space 0) | 390-392 | ignore |
| it does not come from a local array (`AllocaInst`) | 395 | ignore (stays in L1) |
| it does not come from shared memory (`kSharedBaseFn`) | 396-398 | ignore (stays in L1) |
| it has a known per-iteration step, nonzero, fixed in the loop | 399-400 | ignore |
| the distance between lanes is known and nonzero | 406-407 | ignore (unknown or broadcast) |
| if it is constant, it is 64 bytes or more | 408 | ignore (coalesced) |
| the same future address has not yet received a prefetch | 409 | avoids duplicates |
| the formula can be computed at that point | 410 | ignore |

The prefetch is marked as a write (`write ? 1 : 0`) when the access is a store, with maximum locality (3, which becomes `prefetcht0`: brings the line up to the L1D).

### Where it runs in the pipeline

`src/compiler/simt.cpp:2720`

```cpp
    if (!profile_ && insert_prefetches()) uni_ = std::make_unique<Uniformity>(f_, *dt_, *pdt_, *li_);
```

It does not run in the instrumented (profile) variant. If it inserted anything, it redoes the uniformity analysis, because new instructions appeared.

### Vectorization: one prefetch per lane

`src/compiler/simt.cpp:2448-2487` (excerpt)

```cpp
    // Software prefetch: one per 16-lane chunk, at the chunk's first address
    // (a chunk usually spans one or two cache lines).
    if (auto* call = dyn_cast<IntrinsicInst>(&inst); call && call->getIntrinsicID() == Intrinsic::prefetch) {
        Value* ptr = call->getArgOperand(0);
        Function* pf = Intrinsic::getDeclaration(&m_, Intrinsic::prefetch, {ptr->getType()});
        if (!uni_->is_varying(ptr)) {
            b_.CreateCall(pf, {scalar(ptr), call->getArgOperand(1), call->getArgOperand(2), call->getArgOperand(3)});
            return true;
        }
        SplitAddress addr = split_address(ptr);
        // Lanes of a chunk next to each other (known at compile time): one
        // prefetch for the chunk. Spread out (strided per lane, unknown): one
        // per lane, or 15 of the 16 lines would be missed.
        std::vector<char> kind;
        bool known = false;
        if (LinearForm form; linear_form(addr.offsets, form)) {
            std::vector<Value*> firsts;
            kind = static_chunks(form, 4, firsts);
            known = true;
        }
        // Prefetches the compiler added (insert_prefetches) are only there
        // for lanes a cache line or more apart: one per lane.
        if (call->getMetadata("cpusim.auto")) kind.assign(W / 16, 0);
        (void)known;
        for (unsigned h = 0; h < W / 16; ++h) {
            const bool together = h < kind.size() && kind[h] != 0;
            Value* m_bits = b_.CreateBitCast(half(mask, h), b_.getInt16Ty());
            for (unsigned l = 0; l < (together ? 1u : 16u); ++l) {
                // ... one prefetch for lane l, only if it is active
            }
        }
        return true;
    }
```

This is the point that made the manual experiment yield little at first. The comment on the first two lines describes the old behavior (one prefetch per chunk of 16 lanes, at the address of the first); the current code only does this when it can **prove** that the lanes of the chunk are neighbors (`static_chunks`, see [12-static-access-analysis.md](12-static-access-analysis.md)). In the other cases, and whenever the prefetch has the tag `cpusim.auto`, it generates 16 prefetches per register, one per lane. An inactive lane receives a harmless address (the warp's own context, which is already in L1) via select instead of a branch, which keeps the code free of mask-dependent branches.

## When it applies (and when it does not)

- **Only innermost loops.** Outer loops do not receive prefetch.
- **Only global memory.** Shared memory and local arrays stay in L1 and do not need it.
- **Only fixed steps per iteration.** Indirect accesses (`a[b[i]]`, as in SpMV) have no fixed step and are left out.
- **Only spread-out lanes.** This is the filter that weighs the most: most kernels well written for GPUs access memory in a coalesced way (neighboring lanes at neighboring addresses), and for them nothing changes. Automatic prefetch only enters in the "each lane on its own stream" pattern, which is precisely what the hardware prefetchers do not follow.
- **Not in the instrumented variant**, so as not to distort the instruction count.

## Results

| Measurement | Before | After | OpenMP |
|---|---|---|---|
| matrix-rotate, median of 5 alternating runs, 50 launches | 5.45 ms | **3.97 ms** | 4.21 ms |
| matrix-rotate, table with 10 launches | | 4.73 ms | 4.75 ms |

Recall that our side uses 28 CPUs and OpenMP 32. With automatic prefetch, `matrix-rotate` became faster than OpenMP with fewer CPUs.

Among the 9 HeCBench applications we use, **only `matrix-rotate` received prefetches**: 896 prefetch instructions, summing all the generated width variants. The other 8 stayed the same, within measurement noise. This is expected: the spread-out lanes filter leaves out those that are already coalesced. The test suite kept all 18 tests passing.

## Limitations and trade-offs

- **More instructions.** With 16 prefetches per register, the loop gets larger. For `matrix-rotate` it pays off greatly; for a case with few active lanes or very short loops, it might not. The minimum distance of 8 iterations and the lane filter reduce this risk.
- **Fixed distance by formula.** "About 1000 instructions ahead" is a rule of thumb, not a measurement. This machine's memory and the size of the loop body may call for another value in other kernels.
- **Symbolic steps always qualify.** If the distance between lanes is `n * 4` and, in a specific launch, `n` is small (say 4), the lanes would be on the same line and the prefetch would be useless. The compiler does not know `n` and prefers to insert; the cost in that case is only extra instructions.
- **Gather/scatter remains.** The prefetch hides the wait for memory, but each access is still a gather of 16 addresses, more expensive than a contiguous read.

## See also

- [00-basic-concepts.md](../00-basic-concepts.md): cache, cache line, coalescing, gather.
- [06-dynamic-coalescing.md](../iteration-1/06-dynamic-coalescing.md): how contiguous accesses become contiguous reads.
- [07-uniform-loops.md](../iteration-1/07-uniform-loops.md): the uniformity analysis used by `lane_step`.
- [12-static-access-analysis.md](12-static-access-analysis.md): `linear_form` and `static_chunks`, which decide "one prefetch per chunk".
- [16-splitting-blocks-into-warps.md](16-splitting-blocks-into-warps.md): the previous step in `matrix-rotate`.
- [22-what-did-not-work.md](22-what-did-not-work.md): other investigations.
