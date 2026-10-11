# 10. Barrier pass: the last to arrive does not need to wait

> **In one sentence:** when the executor knows that all the other warps of the block are already stopped at a warp's next barrier, it notifies that warp (the `barrier_pass` field in the context), and the warp goes through the barrier directly, without saving state, without returning and without being called again: one context switch less.

The general concepts (warp, `__syncthreads`, warp context) are in [00-basic-concepts.md](../00-basic-concepts.md). The suspension and resumption mechanism at a barrier, which this document presupposes, is in [09-state-at-barriers.md](../iteration-1/09-state-at-barriers.md).

## The problem

On the CPU, a worker executes the warps of a block one at a time. When a warp reaches a `__syncthreads`, it writes its live values to the frame, returns `CPUSIM_BARRIER` to the executor, and is later called again and restores everything through the dispatcher (`resume`). This is a **context switch**: it costs the frame writes, the function return, the new call, the dispatcher's `switch` and the frame reads.

Consider a block with 3 warps and one barrier:

```text
W0: runs to the barrier, saves, returns
W1: runs to the barrier, saves, returns
W2: runs to the barrier, saves, returns      <- last to arrive
W0: restores, continues ...
W1: restores, continues ...
W2: restores, continues ...                   <- just saved what it now restores
```

W2 is the **last** to arrive. At the instant it reaches the barrier, the barrier predicate (all the warps of the block have arrived) already holds: W0 and W1 are stopped at it. Even so, W2 writes its live values to the frame, returns to the executor and, on the next call, goes through the dispatcher and rereads from the frame exactly the values it just wrote. This whole round trip is redundant, because the values were still in registers at the moment of suspension.

## The idea

The warps of a block are executed serially, by a single worker, in an order that the executor itself chooses. Therefore, the executor knows, before each call, how many warps have not yet reached the next barrier of the warp it is about to call. If this number is zero (all the others are already stopped at it), the barrier predicate will be satisfied when the called warp gets there, and the suspension serves no purpose.

The executor communicates this to the generated code through a 4-byte field in the warp context, `barrier_pass`, written before the call. The generated code, on reaching the barrier, tests the field: if it is set, it branches to the block that comes after the barrier, without `save_state`, without `ret` and without going through the dispatcher on the next call.

## How it works, step by step

### Barrier counting per warp

The executor keeps, for each warp `w` of the block, a counter `phase[w]`: how many barriers that warp has already "reached". Selection rule:

1. Find the smallest `phase` among the warps that have not yet finished (`low`) and how many warps have that value (`behind`, the "laggards").
2. Execute the laggards, one at a time.
3. When calling the **last** laggard (`behind == 1`), set `barrier_pass`.
4. After the call: if the warp finished, mark it as done; if it stopped at a barrier, add 1 to `phase` (or 2, if it went through a barrier with the pass and stopped at the next one).

Why is this safe? A warp with `phase = p` is stopped at barrier `p` (or at the start, if `p = 0`) and, when called, will run to the next barrier. For this to be correct, all the other warps must have reached at least barrier `p`; since we always run the one with the **smallest** `phase`, this is guaranteed. And when only one laggard remains, all the others are already at `p + 1` or beyond, that is, they have already stopped at its next barrier: it can go through it.

### A complete example

Block of 3 warps, kernel with 3 barriers (B1, B2, B3). Notation: `phase` of each warp after each call.

```text
                         phase W0  W1  W2   comment
start                          0   0   0    low = 0, behind = 3
call W0 (pass = 0)             1   0   0    stopped at B1
call W1 (pass = 0)             1   1   0    stopped at B1
call W2 (pass = 1)             1   1   2    went through B1, stopped at B2
                                            low = 1, behind = 2 (W0, W1)
call W0 (pass = 0)             2   1   2    resumed from B1, stopped at B2
call W1 (pass = 1)             2   3   2    resumed from B1, went through B2, stopped at B3
                                            low = 2, behind = 2 (W0, W2)
call W0 (pass = 0)             3   3   2
call W2 (pass = 1)             3   3   *    resumed from B2, went through B3, finished
                                            low = 3, behind = 2 (W0, W1)
call W0 (pass = 0)             *   3   *    finished
call W1 (pass = 1)             *   *   *    finished
```

Without the pass there would be 4 calls per warp (one up to each barrier and one to the end), 12 in total. With the pass there were 9: three suspensions avoided. Note that the pass is never used on the **resumption** (the warp that resumes from B1 was already cleared for that); it applies to the next barrier that the warp encounters in the call.

### One pass per call

The `barrier_pass` field concerns the **next** barrier. After going through it, the warp may encounter another barrier in the same call, and at that second one nobody guarantees that the others have arrived. For this reason the generated code keeps a local variable `barrier.pass.used` (it exists only during the call, and never goes to the frame): it starts false, becomes true when the pass is used, and the barrier is only crossed if `barrier_pass != 0` **and** the pass has not yet been used.

## The code

### The field in the contract

`src/device/kernel_abi.h:67-70`

```cpp
    // Set by the runtime when every other warp of the block already reached
    // this warp's next barrier: the warp goes through it instead of
    // suspending (one context switch less). Valid for one barrier per call.
    uint32_t barrier_pass;
```

`src/device/kernel_abi.h:148`

```cpp
#define CPUSIM_CTX_BARRIER_PASS 44
```

The field sits in the context's 128-byte header, at offset 44 (right after `grid_dim`). The generated code uses the constant `CPUSIM_CTX_BARRIER_PASS` to read it, and a `static_assert` in the same file guarantees that the constant and the field's real position agree. Since the change altered the context format, the contract version (`CPUSIM_KERNEL_ABI`) went up to 7 at that point; today it is at 8, because of the fields of the cooperative variant ([15-smt-cooperative-variant.md](15-smt-cooperative-variant.md)).

### The executor loop

`src/device/executor.cpp:167-194`

```cpp
        // Always run a warp that is behind (the lowest barrier count). When
        // it is the last one behind, every other warp already waits at its
        // next barrier, so it may go through that barrier without suspending.
        std::fill(phase.begin(), phase.end(), 0);
        size_t remaining = n;
        while (remaining) {
            uint32_t low = UINT32_MAX;
            size_t behind = 0;
            for (size_t w = 0; w < n; ++w) {
                if (done[w]) continue;
                if (phase[w] < low) low = phase[w], behind = 0;
                if (phase[w] == low) ++behind;
            }
            for (size_t w = 0; w < n && behind; ++w) {
                if (done[w] || phase[w] != low) continue;
                cpusim_warp_ctx* c = ctx.at(w);
                static const bool disabled = std::getenv("CPUSIM_NO_BARRIER_PASS") != nullptr;  // for comparisons
                const bool pass = behind == 1 && !disabled;
                c->barrier_pass = pass;
                --behind;
                if (job.entry(c) == CPUSIM_EXIT) {
                    done[w] = 1;
                    --remaining;
                } else {
                    phase[w] += pass ? 2 : 1;  // went through one barrier, stopped at the next
                }
            }
        }
```

This excerpt is in `Executor::run_blocks`. The first inner loop finds the smallest `phase` and how many warps are at it. The second calls those warps in order, decrementing `behind`; when `behind` is 1, the current warp is the last laggard and receives `pass = true`. The line `phase[w] += pass ? 2 : 1` records that, with the pass, the warp advanced two barriers in a single call. The `CPUSIM_NO_BARRIER_PASS` environment variable turns the mechanism off so the difference can be measured.

### The barrier in the generated code

`src/compiler/simt.cpp:2060-2062`

```cpp
        // Per call, never saved: created after the frame layout.
        pass_used_ = new AllocaInst(Type::getInt1Ty(ctx_), 0, "barrier.pass.used", alloca_block_->getTerminator()->getIterator());
        new StoreInst(ConstantInt::getFalse(ctx_), pass_used_, entry_->getTerminator()->getIterator());
```

The `barrier.pass.used` variable is created **after** the frame layout is decided, on purpose: this way it gets no slot in the frame and is never saved or restored. It starts false at the beginning of each call.

`src/compiler/simt.cpp:2068-2089` (excerpt)

```cpp
void WarpBuilder::emit_barrier(BasicBlock* bb) {
    BasicBlock* resume = BasicBlock::Create(ctx_, "after." + bb->getName(), w_);
    BasicBlock* through = BasicBlock::Create(ctx_, "barrier.through", w_);
    BasicBlock* suspend = BasicBlock::Create(ctx_, "barrier.suspend", w_);
    resumes_.push_back(resume);
    barrier_order_.push_back(bb);
    Value* pass = b_.CreateAlignedLoad(b_.getInt32Ty(),
                                       b_.CreateConstInBoundsGEP1_64(b_.getInt8Ty(), ctx_arg_, CPUSIM_CTX_BARRIER_PASS),
                                       Align(4), "barrier.pass");
    Value* go = b_.CreateAnd(b_.CreateICmpNE(pass, b_.getInt32(0)),
                             b_.CreateNot(b_.CreateLoad(b_.getInt1Ty(), pass_used_)));
    b_.CreateCondBr(go, through, suspend);

    b_.SetInsertPoint(through);
    b_.CreateStore(b_.getTrue(), pass_used_);
    b_.CreateBr(resume);

    b_.SetInsertPoint(suspend);
    save_state(int(resumes_.size()));
    b_.CreateRet(b_.getInt32(CPUSIM_BARRIER));
    b_.SetInsertPoint(resume);
}
```

Each barrier becomes a fork:

```text
                 barrier.pass != 0  and  !pass_used ?
                    /                         \
                 yes                          no
                  |                            |
        barrier.through                 barrier.suspend
        pass_used = true                save_state(k)
                  |                     return CPUSIM_BARRIER
                  v
             after.<barrier>    <---- (on the next call, via resume/restore.k)
```

The `through` path leads directly to the same `after.*` block where the dispatcher would jump on a resumption. The difference is that, through it, the values stay in registers: nothing is written to or read from the frame.

## When it applies (and when it does not)

- **Only in the variants that suspend** (32 lanes, `64s`, `128s`), executed by the `run_blocks` loop. The whole-block variants and the cooperative one do not suspend, so they have nothing to save.
- **Only one warp per round** receives the pass: the last laggard. In a block of 8 warps, each barrier round saves one context switch out of 8. The **fewer** groups per block, the larger the fraction saved (with 2 groups, half of the suspensions disappear).
- **One pass per call**, through the `barrier.pass.used` variable.
- **It does not hold in the optional block function.** There is a `__cpusim_block_` function that runs the whole block with the warps' code embedded; it is only used with `CPUSIM_BLOCK_FN` defined, and the comment in `src/device/device_core.cpp:397-398` records that it does not use `barrier_pass` (the last warp to arrive still suspends) and that it was measured slower than the per-warp calls.
- **Can be disabled** with `CPUSIM_NO_BARRIER_PASS`, for comparison.

## Results

- About **2 to 4% fewer cycles** in `scan` and `bitonic`.
- `scan` with 32 lanes: **36.2 to 36.5 Gcyc** (billions of cycles) with the pass on, against **37.1 to 38.0 Gcyc** off.
- Correct results in both modes.

## Limitations and trade-offs

- **Modest gain by construction.** The pass eliminates at most one suspension per barrier round. In blocks with many warps, most of the switches still exist; to eliminate them, the way is the whole-block variants or the cooperative one.
- **The cost on the common path is small, but not zero.** Every generated barrier now does a 4-byte load from the context (whose line is in L1, since the warp has just read other fields of it), tests a local variable and executes a conditional branch. This is negligible next to a suspension, which writes and rereads the entire live state through the frame and pays for a return, an indirect call and the dispatcher's `switch`.
- **Contract change.** The new field required bumping the ABI version (to 7 at the time), invalidating modules compiled before.

## See also

- [00-basic-concepts.md](../00-basic-concepts.md)
- [02-width-variants.md](../iteration-1/02-width-variants.md): variants that suspend and whole-block variants.
- [09-state-at-barriers.md](../iteration-1/09-state-at-barriers.md): what is saved and restored in a suspension.
- [15-smt-cooperative-variant.md](15-smt-cooperative-variant.md): barrier with no suspension at all, between two SMT threads.
- [16-splitting-blocks-into-warps.md](16-splitting-blocks-into-warps.md): how many groups a block has.
- [17-block-distribution.md](17-block-distribution.md): how the blocks reach each worker.
