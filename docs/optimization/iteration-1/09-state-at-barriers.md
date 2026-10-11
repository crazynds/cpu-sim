# 09. State at barriers: saving only what is needed, and more cheaply

> **In one sentence:** at a `__syncthreads`, the warp function writes the values that are live after the barrier and an identifier of the resume point to a persistent per-warp *frame*, returns to the executor and, on the next call, a dispatcher restores those values and jumps to just after the barrier; the optimizations in this document reduce the saved set (liveness, rematerialization, slot elimination) and the cost of accessing the frame (invariant cache, layout, `!invariant.load`).

The general concepts (warp, lane, variants, shared memory) are in [00-basic-concepts.md](../00-basic-concepts.md).

> **Iteration:** created in iteration 1. In [iteration 2](../iteration-2/README.md), `remat_ok` was added, which also recomputes cheap values that depend on uniform values (the part about `t + threadIdx.y`).

## The problem

In CUDA, `__syncthreads()` is a barrier across all the threads of the block. On the GPU it is cheap: each resident warp has its registers in the SM's register file, and the warp scheduler simply stops issuing instructions from the warps that reached the barrier (the `BAR.SYNC` instruction) until the arrival counter reaches the number of warps in the block. There is no context saving.

On the CPU, a *worker* executes the warps of a block **sequentially**, each warp as a call to the generated function. In a block of 256 threads there are 8 warps of 32 lanes. For warp 0 to get past the barrier, warps 1 to 7 must also reach it; but they only advance when the worker executes them. So warp 0 must **return** in the middle of the kernel and be **resumed** later, at the exact point of the barrier. This is a stackless coroutine transformation, like the one compilers do for `async`/`await`: the function is split into segments between suspension points, and the state that crosses a suspension point goes to explicit memory.

The cost lies in the values that are live across the barrier. They were in registers (vectors in zmm of 64 bytes, 16 floats each) or in the function's stack frame, and both are lost on return. Everything that is read after the barrier must be written to persistent memory before returning and reread on resume.

Example: tiled matrix multiplication ("tiled matmul"). Each thread computes one element `C[row][col]`, walking the inner dimension in tiles copied to shared memory:

```cpp
float acc = 0;
for (int t = 0; t < n; t += TILE) {
    As[ty][tx] = A[row * n + t + tx];
    Bs[ty][tx] = B[(t + ty) * n + col];
    __syncthreads();
    for (int k = 0; k < TILE; ++k) acc += As[ty][k] * Bs[k][tx];
    __syncthreads();
}
C[row * n + col] = acc;
```

In the first version of the compiler, each barrier saved almost everything that was live: `row`, `col`, the address `&As[ty][tx]`, the address `&Bs[ty][tx]`, the indices `ty` and `tx`, masks of active lanes, parameter pointers, and so on. That was about **25 vectors saved per barrier**. With two barriers per loop iteration and eight warps per block, that is about 25 stores of 64 bytes or more per barrier per warp, plus the corresponding loads on resume, all with no useful work.

## The idea

The mechanism has three parts:

- The **resume state**: an `i32` at the start of the frame with the identifier of the suspension point (0 = start of the kernel, `k` = after barrier `k`).
- The **slots**: the live values, written at fixed offsets of the frame. The frame is a **per-warp** region, allocated by the executor, that persists between calls.
- The **dispatcher** (the `resume` block): a `switch` on the state that restores the slots that are live at that point and jumps to the block right after the corresponding barrier.

The optimizations in this document attack the size and the cost of that state:

1. Save only what is live at each resume point (liveness analysis).
2. Do not save what can be recomputed from always-available values (rematerialization).
3. Eliminate redundant slots (copy phis, avoidable copy slots).
4. Compute once per warp, at the start, values that are constant during the block (invariant cache in the frame).
5. Arrange frames, contexts and shared memory so as to avoid set conflicts in L1 (layout).
6. Mark loads of data that is constant during the launch with `!invariant.load`, so the register allocator rematerializes them instead of spilling.

## How it works, step by step

### The suspend and resume cycle

Consider a block with 3 warps (W0, W1, W2) and a kernel with a single barrier:

```text
part A         (each warp computes something and writes to shared)
__syncthreads()
part B         (each warp reads what the others wrote)
```

The executor does:

```text
round 1: calls W0 -> runs A, reaches the barrier, saves state, returns CPUSIM_BARRIER
         calls W1 -> runs A, reaches the barrier, saves state, returns CPUSIM_BARRIER
         calls W2 -> runs A, reaches the barrier, saves state, returns CPUSIM_BARRIER
round 2: calls W0 -> "resume": reads the state (1), restores, runs B, returns CPUSIM_EXIT
         calls W1 -> same
         calls W2 -> same
```

`CPUSIM_BARRIER` and `CPUSIM_EXIT` are the two return codes defined in the contract between the generated code and the runtime (`src/device/kernel_abi.h`). The first indicates suspension at a barrier (the warp must be called again); the second, termination.

On the first call of W0, the state in the frame is 0 (the executor zeroes it before each block). At the barrier, W0 writes the barrier identifier (1, 2, ...) to the state and the live values to its slots in the frame. On the next call the state is different from 0, so the code goes to the `resume` block, which chooses the right restoration with a `switch`.

### The "s" variants

Besides the 32-lane warp, kernels with barriers get "suspending" variants of 64 and 128 lanes (suffix `s`, for example `64s`), which handle 2 or 4 warps at once in a single call. The mechanism is the same: they save to the frame and return `CPUSIM_BARRIER`. The advantage is that, with 4 warps per call, a block of 256 threads has 2 groups instead of 8, so there are 4 times fewer switches per barrier. The disadvantage is that each vector slot becomes 4 times larger (128 lanes instead of 32). Everything this document describes applies equally to them. See [02-width-variants.md](02-width-variants.md).

### Liveness: what to save at each barrier

A variable is **live** at a point *p* if there is a path in the CFG from *p* to a use of it that does not go through a redefinition. It is a backward data-flow problem, solved by fixed-point iteration with `in[B] = use[B] ∪ (out[B] \ def[B])` and `out[B] = ∪ in[S]` over the successors. Example:

```text
x = 5
y = 7
__syncthreads()      <- resume point
y = 9
print(x + y)
```

At the resume point, `x` is live (it is read without redefinition) and must be saved; `y` is dead (it is redefined before any use). The function `compute_resume_liveness` computes this for **each** resume point separately, and each point has its own restore list.

There is a subtlety. The generated code is **linearized**: divergent branches become a sequence of blocks predicated by mask (see [07-uniform-loops.md](07-uniform-loops.md) and [08-if-conversion.md](08-if-conversion.md)). The linearized CFG contains paths that no individual lane takes (for example, executing the `then` and then the `else` of the same `if`), so the liveness computed over it is an over-approximation: a slot may look live because of a use in a block that, for that lane, is masked. For this reason the **value slots** use liveness computed over the original **per-thread** code (`compute_thread_liveness`): each lane follows its own path, linearization preserves each lane's order, and a masked write fully defines the slot for the lane that executes it.

The **masks** (the slots with the active lanes of each block) do not exist in the per-thread code; they only appear in the linearized code. For them, liveness is computed on the linearized CFG, and here another subtlety arises: the suspension block ends in `ret CPUSIM_BARRIER`, with no successors in the CFG, but execution **continues** at the corresponding resume point (via the dispatcher, which does not exist yet when the analysis runs). The compiler adds a virtual edge "suspension block `k` → resume `k`". Without this edge there was a **latent bug**: with two consecutive barriers, a value that was live across both could fail to be saved at the first, because the analysis did not see the path that went through the second.

```text
v = ...
__syncthreads()   (barrier 1)    <- v must be saved here
__syncthreads()   (barrier 2)
uses v
```

Without the edge, from resume 1 the analysis only reached the `ret` of barrier 2 and concluded that `v` was dead. With the edge, it continues to resume 2, finds the use of `v` and keeps the save.

### Rematerialization: recompute instead of storing

**Rematerialization** is recomputing a value at the point of use instead of keeping it live (in a register, a spill or, here, a frame slot). The candidates are values that are **pure functions** of data that is always available to the warp: `threadIdx`, `blockIdx`, the kernel parameters and constants. In the matmul, `row = blockIdx.y * TILE + threadIdx.y` is like this; so is `col`; so is `&As[ty][tx]` (the base address of the shared memory plus an offset computed from `ty` and `tx`).

Storing one of these in the frame costs a vector write at the barrier and a read on resume. Recomputing costs two or three low-latency integer instructions (a `vpmulld` and a `vpaddd`, for example). The function `rematerialize_invariants` does this: instead of such a value having uses in other basic blocks (which, in linearization, requires a slot), it is **cloned** inside each block where it is used. Since the clone does not cross any barrier, it does not enter the saved state.

This is what removed most of the state from the frame: from about 25 vectors saved per barrier in the matmul to **the accumulator `acc` and a few scalars** (such as the counter of the `t` loop).

#### The recent extension: cheap values that depend on uniform values

A **uniform** value is the same in all active lanes of the warp (for example the counter `t` of the loop above). The compiler keeps it scalar, in a general-purpose register, instead of replicating it in a vector. See [07-uniform-loops.md](07-uniform-loops.md).

Consider `t + threadIdx.y`. It is not invariant (it depends on `t`, which changes on each iteration), but it is very cheap and depends only on a uniform value and an invariant. Before, if this value had a use in another basic block, it got a whole vector slot. In the 256-lane variant of the tiled matmul (which does not suspend, but in linearization also carries values between blocks through slots), this meant a vector of 256 integers of 4 bytes, that is, **1 KiB per tile**, stored in an `alloca` on the stack.

The function `remat_ok` widens the rule: a varying value (different between lanes) can be recomputed if it is a cheap integer operation whose leaves are invariant **or uniform**. The uniform leaf (`t`) is not cloned: it is already a scalar value and, by the SSA dominance property (the definition of `t` dominates the original definition, which dominates all its uses), it can be referenced directly in the use block. Only the cheap computation (`+ threadIdx.y`) is redone in the use block.

### Fewer duplicated slots

Two sources of waste were removed:

- **Trivial copies**: the LCSSA (*loop-closed SSA*) form, required by several LLVM loop passes, inserts single-entry phis in the exit blocks of a loop for every value defined in the loop and used outside it. These phis are copies. When the uniform/varying classification of the copy matches that of the source, the copy is eliminated and its slot disappears.
- **Safe reload**: when a varying slot is read in one block and the value read is used in another, the direct solution creates a second slot (a copy) updated with a *blend* under the block's mask. If no path that a lane can take between the read and the use writes to the slot (`reload_is_safe`), it is enough to read the slot again at the use. One slot fewer means one thing fewer to save at barriers.

### Per-warp invariant cache in the frame

Some values depend only on the warp's context and are constant during the execution of the block. Examples: the starting address of each group of 16 lanes in a memory access, and the bits that say whether an access is contiguous or "broadcast" (all lanes reading the same address). See [06-dynamic-coalescing.md](06-dynamic-coalescing.md).

These data are computed **once**, on the warp's first entry (in the `warp.invariants` block), written to the frame and reread as scalars where needed, including after each resume. The function that does this is called `cached`. Since the data never change, they are not on the list of what to save at barriers: they are already in the frame from the start.

### Layout of the frame, the context and the shared memory

Three adjustments to how data sits in memory:

- **Compact context.** Each warp's context (`cpusim_warp_ctx`: block indices, dimensions, pointers) became a **128-byte** header followed by the per-lane data of only the width in use (`threadIdx.x/y/z` of each lane and the mask of real lanes). Before it was **6.3 KB** per warp, with fields 2 KB apart from each other.
- **Odd stride.** The Zen 4 L1D has 32 KiB, 8 ways and 64-byte lines, hence 64 sets; the set index is bits 6 to 11 of the address. Addresses separated by multiples of 4 KiB fall in the same set, and with a 2 KiB stride only two sets are used. If the frames of 8 warps are 4 KiB apart, the same fields of the 8 frames compete for the 8 ways of a single set, and any other data in that set causes evictions. The function `cpusim_stagger` rounds the distance between consecutive frames (and between contexts) up to an **odd** number of 64-byte cache lines, spreading them out.
- **Shared with a per-worker offset.** The two sibling SMT workers (the two logical threads of the same physical core) share the L1D. If the shared memory buffers of both started page-aligned, the same indices of both would fall in the same sets. Each worker starts its buffer at a different offset within the page.

The frame size is also recorded **per variant** (32, 64s and 128s have different sizes), so the executor reserves only what is needed.

### Invariant reads

Context fields and kernel parameters are constant during the whole launch. The compiler marks these loads with `!invariant.load` (and the parameters pointer with `dereferenceable(N)`). The metadata guarantees that the memory location does not change while it is accessible, so the load is trivially rematerializable for the register allocator: under register pressure, instead of spilling the value to the stack (store + later load), it reissues the load from the original location, saving the store. `dereferenceable` also allows the load to be moved before conditions (safe speculation). The frame is left out of this marking, because it changes between calls.

## The code

### Entry: start or resume

`src/compiler/simt.cpp:2007-2018`

```cpp
    // With barriers, a warp either starts (state 0) or resumes after one.
    if (!barrier_blocks_.empty()) {
        frame_ = b_.CreateAlignedLoad(ptr, field(CPUSIM_CTX_FRAME), Align(8), "frame");
        state_ptr_ = frame_;
        Value* state = b_.CreateAlignedLoad(i32, state_ptr_, Align(64), "state");
        BasicBlock* init = BasicBlock::Create(ctx_, "start", w_);
        dispatch_ = BasicBlock::Create(ctx_, "resume", w_);
        cache_block_ = BasicBlock::Create(ctx_, "warp.invariants", w_);
        b_.CreateCondBr(b_.CreateICmpEQ(state, b_.getInt32(0)), cache_block_, dispatch_);
        BranchInst::Create(init, cache_block_);
        b_.SetInsertPoint(init);
    }
```

The code reads the frame pointer from the context and, in the first `i32` of the frame, the resume state. If the state is 0, the warp is starting: it first goes through `warp.invariants` (where the invariant cache is filled) and then through `start`. If it is not 0, it goes to `resume`, the dispatcher.

### The barrier: save and return

`src/compiler/simt.cpp:2066-2089`

```cpp
// Stores every slot for now; compute_resume_liveness() later removes the
// stores of slots the code after the barrier never reads before writing.
void WarpBuilder::emit_barrier(BasicBlock* bb) {
    BasicBlock* resume = BasicBlock::Create(ctx_, "after." + bb->getName(), w_);
    BasicBlock* through = BasicBlock::Create(ctx_, "barrier.through", w_);
    BasicBlock* suspend = BasicBlock::Create(ctx_, "barrier.suspend", w_);
    resumes_.push_back(resume);
    barrier_order_.push_back(bb);
    // ...
    b_.SetInsertPoint(suspend);
    save_state(int(resumes_.size()));
    b_.CreateRet(b_.getInt32(CPUSIM_BARRIER));
    b_.SetInsertPoint(resume);
}
```

For each `__syncthreads`, the compiler creates a suspension block (which saves the state and returns `CPUSIM_BARRIER`) and an `after.*` block where execution continues afterwards. As the comment above says, at first **all** slots are saved; the liveness analysis later deletes those that are not needed. The omitted part (`// ...`) is the `barrier.through` path, explained in [10-barrier-pass.md](../iteration-2/10-barrier-pass.md).

`src/compiler/simt.cpp:2091-2103`

```cpp
void WarpBuilder::save_state(int resume_id) {
    std::vector<Save> saves;
    for (auto [a, offset] : state_) {
        Type* t = a->getAllocatedType();
        Value* p = b_.CreateConstInBoundsGEP1_64(b_.getInt8Ty(), frame_, offset);
        LoadInst* ld = b_.CreateLoad(t, a);
        StoreInst* st = b_.CreateAlignedStore(ld, p, Align(std::min<uint64_t>(64, MinAlign(offset, 64))));
        saves.push_back({a, offset, ld, st});
    }
    b_.CreateAlignedStore(b_.getInt32(resume_id), state_ptr_, Align(64));
    saves_.push_back(std::move(saves));
    save_blocks_.push_back(b_.GetInsertBlock());
}
```

Each slot is copied to its fixed offset in the frame, and last the barrier identifier is written to the state. The (read, write) pair of each slot is kept in `saves_` precisely so that the liveness analysis can delete it later.

### The dispatcher

`src/compiler/simt.cpp:2233-2253`

```cpp
void WarpBuilder::emit_dispatch() {
    compute_resume_liveness();
    b_.SetInsertPoint(dispatch_);
    Value* state = b_.CreateAlignedLoad(b_.getInt32Ty(), state_ptr_, Align(64));
    BasicBlock* bad = BasicBlock::Create(ctx_, "resume.bad", w_);
    SwitchInst* sw = b_.CreateSwitch(state, bad, unsigned(resumes_.size()));
    for (size_t k = 0; k < resumes_.size(); ++k) {
        BasicBlock* restore = BasicBlock::Create(ctx_, "restore." + Twine(k + 1), w_);
        sw->addCase(b_.getInt32(uint32_t(k + 1)), restore);
        b_.SetInsertPoint(restore);
        for (size_t i = 0; i < state_.size(); ++i) {
            if (!live_at_resume_[k][i]) continue;
            auto [a, offset] = state_[i];
            Value* p = b_.CreateConstInBoundsGEP1_64(b_.getInt8Ty(), frame_, offset);
            b_.CreateStore(b_.CreateAlignedLoad(a->getAllocatedType(), p, Align(std::min<uint64_t>(64, MinAlign(offset, 64)))), a);
        }
        b_.CreateBr(resumes_[k]);
    }
    b_.SetInsertPoint(bad);
    b_.CreateUnreachable();
}
```

A `switch` on the state: for each barrier `k` there is a `restore.k` block that rereads **only** the slots that are live at that resume (`live_at_resume_[k][i]`) and jumps to the code right after the barrier.

### Liveness on the linearized code, with the barrier edge

`src/compiler/simt.cpp:2175-2197`

```cpp
    std::vector<BasicBlock*> order;
    for (BasicBlock* bb : post_order(&w_->getEntryBlock())) order.push_back(bb);
    for (BasicBlock* r : resumes_)  // not reachable from the entry until the dispatch exists
        for (BasicBlock* bb : post_order(r)) order.push_back(bb);
    bool changed = true;
    while (changed) {
        changed = false;
        for (BasicBlock* bb : order) {
            BitVector out(n);
            for (BasicBlock* s : successors(bb)) out |= in[s];
            // A barrier's return continues, later, at its resume point.
            for (size_t k = 0; k < save_blocks_.size(); ++k)
                if (save_blocks_[k] == bb) out |= in[resumes_[k]];
            BitVector cur = def[bb];
            cur.flip();
            cur &= out;
            cur |= use[bb];
            if (cur != in[bb]) {
                in[bb] = cur;
                changed = true;
            }
        }
    }
```

This is the classic liveness analysis (`in = use ∪ (out \ def)`), iterated to the fixed point; post-order speeds up the convergence of a backward problem. Lines 2066-2067 are the fix for the latent bug: the block that saves the state at barrier `k` is treated as if it had resume `k` as a successor.

### Combining with per-thread liveness

`src/compiler/simt.cpp:2199-2228` (excerpt)

```cpp
    // Demoted slots: the per-thread answer is exact; masks only exist here.
    compute_thread_liveness();
    DenseMap<const AllocaInst*, const AllocaInst*> original;  // warp slot -> per-thread slot
    for (auto& [orig, slot] : slots_) original[slot] = orig;
    for (size_t k = 0; k < resumes_.size(); ++k) {
        BitVector live = in[resumes_[k]];
        const auto& thread_live = live_after_barrier_[barrier_order_[k]];
        for (auto& [a, offset] : state_) {
            auto it = original.find(a);
            if (it != original.end() && !thread_live.contains(it->second)) live.reset(index[a]);
        }
        // ...
        live_at_resume_.push_back(live);
        for (Save& s : saves_[k]) {
            if (live[index[s.slot]]) continue;
            Value* addr = s.store->getPointerOperand();
            s.store->eraseFromParent();
            s.load->eraseFromParent();
            if (auto* gep = dyn_cast<Instruction>(addr); gep && gep->use_empty()) gep->eraseFromParent();
        }
    }
```

For the slots that came from values of the per-thread code, the per-thread answer prevails: if there the value is not live after the barrier, it leaves the list (`live.reset`). The masks keep the answer from the linearized code. At the end, the save load/store pairs of dead slots are deleted. The omitted part prints a diagnostic when the environment variable `CPUSIM_DEVC_DEBUG` is set.

### Rematerialization: what can be recomputed

`src/compiler/simt.cpp:678-696`

```cpp
// Like is_invariant, but uniform values may be leaves too: a cheap varying
// value such as `t + threadIdx.y` (t a loop counter) is recomputed in the block
// that uses it from the scalar t, instead of keeping a whole vector in a slot.
bool WarpBuilder::remat_ok(Value* v, unsigned depth) {
    if (is_invariant(v, 0)) return true;
    auto* inst = dyn_cast<Instruction>(v);
    if (!inst || depth > 8) return false;
    if (!uni_->is_varying(inst)) return depth > 0;
    if (!(isa<BinaryOperator>(inst) || isa<CastInst>(inst))) return false;
    switch (inst->getOpcode()) {
        case Instruction::UDiv: case Instruction::SDiv: case Instruction::URem: case Instruction::SRem:
        case Instruction::FDiv: case Instruction::FRem:
            return false;
        default:
            break;
    }
    if (inst->getType()->isFloatingPointTy()) return false;
    return all_of(inst->operands(), [&](Use& u) { return remat_ok(u.get(), depth + 1); });
}
```

Invariants are always accepted. A uniform value can be a **leaf** (`depth > 0`), but not the root (a scalar that is already available has nothing to recompute). A varying value only qualifies if it is an **integer** arithmetic or conversion operation: floating point is excluded (higher latencies and, in cases such as contraction to FMA, a risk of different results); divisions are excluded (high latency and the possibility of `#DE` where the original would not execute). The depth is limited to 8 to avoid cloning long chains.

### Rematerialization: cloning in the use block

`src/compiler/simt.cpp:698-732` (excerpt)

```cpp
void WarpBuilder::rematerialize_invariants() {
    // ...
    std::function<Value*(Value*, BasicBlock*)> clone_into = [&](Value* v, BasicBlock* bb) -> Value* {
        auto* inst = dyn_cast<Instruction>(v);
        if (!inst || inst->getParent() == bb) return v;
        // A uniform leaf of a remat_ok value stays where it is (SSA: it
        // dominates the original, so it dominates the use too).
        if (!is_invariant(inst, 0) && !uni_->is_varying(inst)) return v;
        if (auto it = placed.find({bb, v}); it != placed.end()) return it->second;
        Instruction* c = inst->clone();
        for (Use& op : c->operands())
            if (!isa<Function>(op.get())) op.set(clone_into(op.get(), bb));
        c->setName(inst->getName() + ".remat");
        c->insertBefore(anchor[bb]->getIterator());
        // ...
        placed[{bb, v}] = c;
        return c;
    };

    std::vector<Instruction*> candidates;
    for (BasicBlock& bb : f_)
        for (Instruction& inst : bb)
            if (is_invariant(&inst, 0) || (uni_->is_varying(&inst) && remat_ok(&inst, 0))) candidates.push_back(&inst);
    for (Instruction* inst : candidates) {
        SmallVector<Use*, 8> far;
        for (Use& u : inst->uses())
            if (cast<Instruction>(u.getUser())->getParent() != inst->getParent()) far.push_back(&u);
        for (Use* u : far) u->set(clone_into(inst, cast<Instruction>(u->getUser())->getParent()));
    }
}
```

For each candidate, every use in another basic block now uses a clone made in the use's own block, with the operands also cloned recursively. The `placed` map avoids cloning the same thing twice in the same block. Note the line that leaves the uniform leaf where it is: it is what allows `t + threadIdx.y` to be recomputed without trying to clone `t`.

### Safe reload instead of a copy slot

`src/compiler/simt.cpp:614-624` (beginning)

```cpp
// True if no path that a single lane can take from `ld` to `use` (without
// executing `ld` again) stores to `slot`.
bool WarpBuilder::reload_is_safe(LoadInst* ld, AllocaInst* slot, Instruction* use) {
    BasicBlock* from = ld->getParent();
    BasicBlock* to = use->getParent();
    auto stores_in = [&](BasicBlock* bb, BasicBlock::iterator begin, BasicBlock::iterator end) {
        for (auto it = begin; it != end; ++it)
            if (auto* st = dyn_cast<StoreInst>(&*it); st && st->getPointerOperand() == slot) return true;
        return false;
    };
    if (stores_in(from, std::next(ld->getIterator()), from->end())) return false;
```

The rest of the function (up to line 645) finds the blocks that lie on some path from the read to the use and checks that none of them writes to the slot. If it is safe, the use rereads the slot directly.

### Invariant cache in the frame

`src/compiler/simt.cpp:1230-1247`

```cpp
// The value of `v`, from the warp's frame if it is warp-invariant.
Value* WarpBuilder::cached(Value* v) {
    if (!cache_block_ || isa<Constant>(v)) return v;
    DenseMap<Value*, Value*> memo;
    Value* at_start = clone_invariant(v, memo);
    if (!at_start || at_start == v) return v;
    const DataLayout& dl = m_.getDataLayout();
    Type* t = v->getType();
    uint64_t align = std::min<uint64_t>(64, dl.getPrefTypeAlign(t).value());
    uint64_t offset = alignTo(cache_end_, align);
    cache_end_ = offset + dl.getTypeAllocSize(t);
    note_frame_size();
    IRBuilder<> sb(cache_block_->getTerminator());
    sb.CreateAlignedStore(at_start, sb.CreateConstInBoundsGEP1_64(sb.getInt8Ty(), frame_, offset), Align(align));
    LoadInst* ld = b_.CreateAlignedLoad(t, b_.CreateConstInBoundsGEP1_64(b_.getInt8Ty(), frame_, offset), Align(align),
                                        v->getName() + ".cached");
    return ld;
}
```

`clone_invariant` tries to rebuild the value in the `warp.invariants` block using only the context and the parameters. If it succeeds, the value gets a place in the frame (after the slots), is written once at the start of the warp, and at the point of use becomes a simple read (`.cached`). Without barriers (`cache_block_` null), nothing changes.

### Invariant reads

`src/compiler/simt.cpp:1990-2005`

```cpp
    // Everything read so far (context fields, kernel parameters) is constant
    // for the whole launch. Saying so lets the register allocator reload such
    // values where needed instead of spilling them at every entry.
    MDNode* empty = MDNode::get(ctx_, {});
    for (Instruction& inst : *entry) {
        auto* ld = dyn_cast<LoadInst>(&inst);
        if (!ld) continue;
        // Only the context and the parameters; never the frame (it changes
        // between calls, and within a block function between rounds).
        Value* base = getUnderlyingObject(ld->getPointerOperand());
        if (base != ctx_arg_ && base != params) continue;
        ld->setMetadata(LLVMContext::MD_invariant_load, empty);
        if (ld->getType()->isPointerTy() && ld == params)
            ld->setMetadata(LLVMContext::MD_dereferenceable,
                            MDNode::get(ctx_, ConstantAsMetadata::get(b_.getInt64(std::max<uint64_t>(k_.param_bytes, 1)))));
    }
```

All reads made in the prologue whose address comes from the context or the parameters receive the `!invariant.load` mark. The parameters pointer also gets `dereferenceable` with the size of the parameter block.

### Layout: odd stride and offset shared memory

`src/device/kernel_abi.h:135-140`

```cpp
// Distance between consecutive contexts or frames of one block: an odd number
// of cache lines, so the warps' data does not pile up in the same L1 sets.
static inline uint32_t cpusim_stagger(uint32_t bytes) {
    uint32_t lines = (bytes + 63) / 64;
    return (lines | 1u) * 64;
}
```

It converts the size to 64-byte cache lines and sets the lowest bit (`| 1u`), which forces an odd number. A frame of 4 lines ends up 5 lines from the next; one of 7 stays at 7.

`src/device/executor.cpp:56-62`

```cpp
    // Shared memory of the block this thread runs; first touched here, so it
    // lands in this core's NUMA node and stays hot in its caches.
    // Each worker's buffer starts at a different offset inside its page: the two
    // SMT siblings of a core share one L1, and page-aligned buffers would all
    // land in the same cache sets.
    auto* shared_mem = static_cast<uint8_t*>(std::aligned_alloc(4096, kMaxSharedBytes + 4096));
    uint8_t* shared = shared_mem + (uint64_t(index) * 1088) % 4096 / 64 * 64;
```

The worker with index `index` shifts the start of its shared buffer by `index * 1088` bytes (modulo 4096, rounded to a multiple of 64). This way, neighboring workers (and in particular the two SMT siblings) do not start in the same L1 set.

## When it applies (and when it does not)

- **Only in kernels with `__syncthreads`.** Without barriers, there is no frame, no dispatcher and no saving.
- **Only in the variants that suspend:** the 32-lane one and the `64s` and `128s` ones. The "whole block" variants (64 to 512 lanes, when the block fits in one call) execute all threads in the same call, so the barrier is just a point in the linearized code and there is no frame. The cooperative variant (`c`) also uses no frame: the barrier is a real synchronization between the two sibling SMT workers ([15-smt-cooperative-variant.md](../iteration-2/15-smt-cooperative-variant.md)).
- **Rematerialization (including `remat_ok`) applies to all variants**, including those that do not suspend: it reduces slots, which even without a barrier become variables on the stack (this is the case of the 1 KiB per tile example in the 256-lane variant).
- **What is not rematerialized:** divisions and remainders (high latency and possible exception), floating-point operations when the value is varying, loads (memory may have been written between the definition and the use, including by other warps across the barrier) and chains with depth above 8.
- **`reload_is_safe` only for varying slots.** In a uniform slot, a scalar store changes the value of all lanes at once, so the per-lane argument (each lane only observes stores made under its own mask) does not hold.
- **`!invariant.load` never on the frame**, because the frame changes between calls (and, in the optional block function `__cpusim_block_`, between rounds).

## Results

- In the tiled matmul, rematerialization reduced the state saved per barrier from **about 25 vectors to the accumulator and a few scalars**.
- The per-warp context went from **6.3 KB** to a **128-byte** header plus the per-lane data of the width in use.
- The `remat_ok` extension avoids storing a **1 KiB per tile** vector in a stack `alloca` in the 256-lane variant of the tiled matmul.
- The fix of the "barrier to resume" edge is about **correctness**, not speed: it removed a latent bug with two consecutive barriers.

The remaining changes (fewer duplicated slots, invariant cache, odd stride, offset shared memory, invariant reads) have no isolated measurement recorded; they were made together and evaluated by the overall effect on the benchmarks with barriers.

## Limitations and trade-offs

- **Recomputing is not always cheaper.** The `remat_ok` rule is deliberately conservative (integers only, no division, depth up to 8). Cloning a long chain in several blocks could cost more than a read from the frame.
- **The frame still exists.** Values that are really live, such as the accumulator, are still written and read at each barrier. To eliminate even that in some cases, there is the shortcut in [10-barrier-pass.md](../iteration-2/10-barrier-pass.md); to eliminate suspension altogether, the whole-block variants ([02-width-variants.md](02-width-variants.md)) and the cooperative one ([15-smt-cooperative-variant.md](../iteration-2/15-smt-cooperative-variant.md)).
- **The invariant cache takes up frame space.** It trades recomputation on each resume for a memory read; the frame gets slightly larger.
- **Two liveness analyses.** Maintaining one over the per-thread code and another over the linearized code is more complex, but it is the price of being exact for both kinds of slot.

## See also

- [00-basic-concepts.md](../00-basic-concepts.md): warps, lanes, barriers, shared memory.
- [02-width-variants.md](02-width-variants.md): 32-lane, "s" and whole-block variants.
- [06-dynamic-coalescing.md](06-dynamic-coalescing.md): the access pattern bits stored in the invariant cache.
- [07-uniform-loops.md](07-uniform-loops.md): uniform values and loops with real control flow.
- [08-if-conversion.md](08-if-conversion.md): linearization and masks.
- [10-barrier-pass.md](../iteration-2/10-barrier-pass.md): crossing the barrier without suspending.
- [15-smt-cooperative-variant.md](../iteration-2/15-smt-cooperative-variant.md): real barrier between two SMT threads, without a frame.
- [16-splitting-blocks-into-warps.md](../iteration-2/16-splitting-blocks-into-warps.md): how the block is split into groups.
