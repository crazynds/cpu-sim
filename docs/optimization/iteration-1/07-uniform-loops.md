# 07. Uniform loops with real control flow

> **In one sentence:** when the compiler proves that all the threads of a warp take the same number of trips through a loop (as in `for (k = 0; k < K; ++k)`), it generates the loop as an ordinary CPU loop, with plain branches, instead of using the mask machinery that is only needed when threads can split apart.

## The problem

### Divergence

In the SIMT model, the 32 threads of a warp share a single instruction stream. In our compiler, each thread is a **lane** of an AVX-512 vector (the general concepts are in [00-basic-concepts.md](../00-basic-concepts.md)). As long as all lanes follow the same path, the vector code is straightforward. The case that needs handling is a branch whose condition differs between lanes:

```cpp
if (x[i] > 0) y[i] = 1; else y[i] = 2;
```

This is **divergence**: lanes of the same warp take different sides of the branch. Since there is a single program counter per warp, the solution (on a GPU, in hardware; here, in the generated code) is to **execute both paths** in sequence, each under a predication **mask** holding the lanes that are active on that path. Inactive lanes pass through the code, but their effects (stores, writes to live registers) are suppressed by the mask (in AVX-512, `k` registers and masked operations).

Example with a 4-lane warp, `x = (5, -3, 7, -1)`:

```text
lane:                   0     1     2     3
x:                      5    -3     7    -1
condition x > 0:        T     F     T     F

"then" block (y = 1), mask 1 0 1 0:   y = (1, ?, 1, ?)
"else" block (y = 2), mask 0 1 0 1:   y = (1, 2, 1, 2)
```

### Linearization

Transforming the kernel's **CFG** (control flow graph) into this sequence of predicated basic blocks is called **linearization**: the CFG with branches is replaced by a fixed order of blocks, and the choice of path stops being a jump and becomes a per-block mask.

In linearization, each basic block:

1. reads its mask (the lanes that reached it);
2. tests whether **any** lane is active; if none is, it skips the block;
3. executes the instructions under that mask;
4. at the end, computes the mask of each successor (`mask AND condition` for one side, `mask AND NOT condition` for the other) and **adds** (OR) those masks to the successors' stored masks.

For a loop, linearization is more expensive. Each lane may have a different number of iterations, so on every iteration the code must: zero the masks of the inner blocks, execute the body, accumulate into a "next-iteration mask" the lanes that took the back edge, and test whether that mask still has any lane. The loop only ends when the last lane exits.

### The cost of treating everything this way

The first version of the compiler linearized **every** branch this way, including those that did not need it. The loop of the naive matrix multiplication is the typical case:

```cpp
for (int k = 0; k < K; ++k) acc += A[row * K + k] * B[k * N + col];
```

`k` starts at 0 for all threads, grows by 1 for all of them and is compared with `K`, which is the same for all. **All threads take exactly the same trips.** Even so, each iteration loaded and stored masks, tested "is any lane still active?" and reinitialized the body masks. In a short body like this one (two loads and an FMA), this overhead had a cost comparable to that of the useful work.

## The idea

A value is **uniform** if it is the same in all active lanes, and **varying** if it can differ. This classification is done by the uniformity analysis (`src/compiler/uniformity.cpp`), a data-flow analysis that starts from varying sources (`threadIdx`, loads from varying addresses) and propagates variability through data dependences and control dependences (a phi at the join point of a varying branch is also varying). A branch is **uniform** if its condition is uniform: all active lanes go to the same side.

If **all** the branches of a loop are uniform (the exit test, the jump back to the start, the inner `if`s and those of nested loops), then all the lanes that entered the loop follow exactly the same path. The mask does not change anywhere: it is the same as at the loop entry. The loop can be generated **with its original CFG and scalar branches**, as in an ordinary CPU program.

## How it works, step by step

Let us follow the loop `for (k = 0; k < 3; ++k) acc += ...` with the 4-lane warp, in which lane 3 is not active (for example, it exited earlier in an `if (row >= M) return;`). The entry mask is `1 1 1 0`.

**Linearized (before):**

```text
entry:    header_mask = 1110
trip 1:   zero the body masks
          body with mask 1110
          condition k < 3 per lane: (T T T ?) -> next = 1110
          any lane in "next"? yes
trip 2:   zero the body masks; body; next = 1110; any? yes
trip 3:   zero the body masks; body; next = 1110; any? yes
trip 4:   condition: (F F F ?) -> next = 0000; any? no -> exit
```

**Uniform loop (after):**

```text
entry:    m = 1110; any lane active? yes
trip 1:   body with m;  k < 3 (a single number)? yes, loop back
trip 2:   body with m;  k < 3? yes, loop back
trip 3:   body with m;  k < 3? yes, loop back
          k < 3? no -> exit; the lanes in m continue to after the loop
```

What disappeared from each trip: the zeroing of masks, the mask operations at the end of the body and the "any lane?" test. What is left is a scalar `k < 3` and an ordinary conditional branch (`cmp` + `jcc`), which the branch predictor gets right on every iteration except the last.

### Steps

1. **Detect:** for each loop, look at all the blocks it contains (including those of sub-loops) and check that every conditional branch has a uniform condition.
2. **Generate:** create a copy of each block of the loop, with the same edges as the original CFG. The instructions of each block use the loop's **entry mask** (it is still needed: lanes that never entered the loop, like lane 3 above, must not write to memory).
3. **Exit:** when a branch leads out of the loop, the lanes (all together) are added to the mask of the destination block, and generation continues with the code after the loop, which goes back to being linearized normally.

### Loops with a barrier

A `__syncthreads()` (barrier) inside the loop complicates things: the warp's execution **stops** at the barrier and comes back later, possibly in the middle of the loop (see [09-state-at-barriers.md](09-state-at-barriers.md)). On return, the variable that held the entry mask no longer exists. The solution is to keep the mask in the loop's **header slot** (the memory location that already holds that block's mask and that is part of the warp state preserved across suspensions) and to reread that slot in each block. This way, loops such as the `for (t = 0; t < K; t += TILE)` of the shared-memory multiplication, which has two barriers inside, are also generated as uniform loops.

## The code

First, for comparison, the end of an ordinary linearized block, where the successors' masks are computed:

`src/compiler/simt.cpp:2294-2312`

```cpp
    if (auto* br = dyn_cast<BranchInst>(bb->getTerminator())) {
        if (br->isUnconditional()) {
            add_edge(bb, br->getSuccessor(0), mask);
        } else {
            Value* c = br->getCondition();
            Value *taken, *not_taken;
            if (uni_->is_varying(c)) {
                Value* cv = vector(c);
                taken = b_.CreateAnd(mask, cv);
                not_taken = b_.CreateAnd(mask, b_.CreateNot(cv));
            } else {
                Value* cs = scalar(c);
                taken = b_.CreateSelect(cs, mask, no_lanes());
                not_taken = b_.CreateSelect(cs, no_lanes(), mask);
            }
            add_edge(bb, br->getSuccessor(0), taken);
            add_edge(bb, br->getSuccessor(1), not_taken);
        }
    }
```

Even with a uniform condition (`else`), the linearized block still chooses between "all lanes" and "none" and adds the result to the masks of both successors with `add_edge`. For a loop, this happens on every trip.

The detection of the uniform loop:

`src/compiler/simt.cpp:2320-2327`

```cpp
bool WarpBuilder::is_uniform_loop(Loop* l) {
    for (BasicBlock* bb : l->blocks()) {
        if (auto* br = dyn_cast<BranchInst>(bb->getTerminator()); br && br->isConditional() &&
                                                                    uni_->is_varying(br->getCondition()))
            return false;
    }
    return true;
}
```

`l->blocks()` includes the blocks of sub-loops. A single conditional branch with a varying condition is enough for the loop not to be uniform.

The generation:

`src/compiler/simt.cpp:2332-2359`

```cpp
void WarpBuilder::emit_uniform_loop(Loop* l) {
    BasicBlock* header = l->getHeader();
    std::string name = header->getName().str();
    bool barriers = any_of(l->blocks(), [&](BasicBlock* bb) { return barrier_blocks_.contains(bb); });
    Value* entry_mask = b_.CreateLoad(mask_type_, block_mask_[header], "m.uloop." + name);
    BasicBlock* exit = BasicBlock::Create(ctx_, "uloop." + name + ".exit", w_);
    DenseMap<BasicBlock*, BasicBlock*> emitted;
    for (BasicBlock* bb : l->blocks()) emitted[bb] = BasicBlock::Create(ctx_, "u." + bb->getName(), w_);
    b_.CreateCondBr(any_lane(entry_mask), emitted[header], exit);

    for (BasicBlock* bb : l->blocks()) {
        b_.SetInsertPoint(emitted[bb]);
        // With barriers inside, a resume enters the loop in the middle: the
        // mask then comes back from the header's slot (restored with the rest).
        Value* mask = barriers ? b_.CreateLoad(mask_type_, block_mask_[header], "m.u") : entry_mask;
        bool divergent = uni_->is_divergent_block(bb);
        uint64_t counts[CPUSIM_COUNTERS] = {};
        for (Instruction& inst : *bb) {
            if (profile_) count(inst, counts);
            if (inst.isTerminator()) break;
            if (!emit_instruction(inst, mask, divergent)) return;
        }
        if (profile_) emit_counters(counts, mask);

        if (barrier_blocks_.contains(bb)) {
            emit_barrier(bb);
            mask = b_.CreateLoad(mask_type_, block_mask_[header], "m.u");
        }
```

- `entry_mask` is read once, from the header slot. A single `any_lane` test decides whether the loop runs: if no lane arrived, it jumps straight to the exit.
- `emitted` creates a new block for each block of the loop. The edges between them will be those of the original CFG.
- Inside each block, `mask` is the entry mask or, with barriers, a reread of the header slot. After a barrier (`emit_barrier`), the mask is reread again, because the warp may have been suspended and resumed there.
- `profile_` is the instrumented variant that counts instructions; it also uses this generation.

And the end of each block, with scalar branches:

`src/compiler/simt.cpp:2361-2381`

```cpp
        // Leaving the loop: the lanes join the target's mask, then continue
        // with the code after the loop.
        auto target = [&](BasicBlock* s) -> BasicBlock* {
            if (l->contains(s)) return emitted[s];
            BasicBlock* edge = BasicBlock::Create(ctx_, "u.exit." + s->getName(), w_);
            IRBuilder<>::InsertPointGuard guard(b_);
            b_.SetInsertPoint(edge);
            add_edge(bb, s, mask);
            b_.CreateBr(exit);
            return edge;
        };
        Instruction* term = bb->getTerminator();
        if (auto* br = dyn_cast<BranchInst>(term)) {
            if (br->isUnconditional())
                b_.CreateBr(target(br->getSuccessor(0)));
            else
                b_.CreateCondBr(scalar(br->getCondition()), target(br->getSuccessor(0)), target(br->getSuccessor(1)));
        } else {
            b_.CreateBr(exit);  // return / unreachable: these lanes are done
        }
    }
```

- If the destination is inside the loop, the branch goes to its copy: it is an ordinary jump.
- If it is outside, it creates a small exit block that performs the single `add_edge` (the lanes join the destination's mask) and continues to the code after the loop.
- `CreateCondBr(scalar(...))` is the central point: the condition is a single boolean, and the branch is a normal CPU jump instruction.

What chooses between the two modes:

`src/compiler/simt.cpp:2385-2386`

```cpp
void WarpBuilder::emit_loop(Loop* l) {
    if (is_uniform_loop(l)) return emit_uniform_loop(l);
```

## When it applies (and when it does not)

- **It applies** when all the conditional branches of the loop, including sub-loops, have a uniform condition. In practice: loops whose counter starts at a uniform value, advances uniformly and ends at a uniform bound (kernel parameters, `blockDim`, constants). This is the case for most inner loops in CUDA (matrix multiplication, stencils). Note: the loop must be uniform as a whole. The classic reduction `for (s = blockDim.x / 2; s > 0; s >>= 1) { if (tid < s) ...; __syncthreads(); }` has a uniform counter, but the inner `if (tid < s)` is varying and contains a store, so this loop remains linearized.
- **It does not apply** if some branch inside depends on the thread. Examples: `for (i = threadIdx.x; i < n; i += blockDim.x)` (each thread ends on a different trip) or an `if (x[i] > 0)` inside the body. In these cases the whole loop goes back to being linearized. A small varying `if` inside the loop can be eliminated beforehand by [if-conversion](08-if-conversion.md), which sometimes makes the loop uniform.
- **Loops with a barrier** are accepted, at the price of rereading the mask from the slot in each block.
- **The mask still exists:** the lanes that did not enter the loop stay off throughout the body. The gain is not having to **update** the mask.

## Results

When this optimization landed, the naive matmul went from **912 ms to 597 ms** (about 35% less time). There is no isolated measurement for other kernels.

## Limitations and trade-offs

- The test is all or nothing: a single varying branch anywhere in the loop (even in a rare sub-loop) causes the whole loop to be linearized.
- Uniformity is decided at compile time, conservatively. A loop that is uniform in practice, but whose condition the compiler cannot prove uniform (for example, a bound read from a thread-indexed array that happens to be equal for all threads), remains linearized.
- With barriers, each block rereads the mask from memory. It is a load that hits in L1, but it is still one more instruction per block.

## See also

- [00-basic-concepts.md](../00-basic-concepts.md): lane, warp, mask, uniform and varying.
- [08-if-conversion.md](08-if-conversion.md): eliminating small divergent branches.
- [09-state-at-barriers.md](09-state-at-barriers.md): what is saved when the warp stops at a barrier.
- [10-barrier-pass.md](../iteration-2/10-barrier-pass.md): crossing barriers without suspending.
- [05-address-decomposition.md](05-address-decomposition.md): what else happens inside the `k` loop of the multiplication.
