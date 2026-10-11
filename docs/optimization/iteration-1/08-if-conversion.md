# 08. SIMT if-conversion with guarded loads

> **In one sentence:** a small `if` on which threads may disagree (such as `x = cond ? A[i] : 0`) is rewritten as straight-line code, with no branch: all lanes compute the side of the `if`, the memory read only happens in the lanes where the condition holds, and at the end a per-lane "pick one or the other" replaces the branch.

## The problem

The document [07-uniform-loops.md](07-uniform-loops.md) explains **divergence**: when the threads of a warp want to follow different paths in an `if`, the compiler executes the paths one after the other, each with a **mask** of active lanes. This transformation (the **linearization**) is correct, but it has a fixed cost per block: loading the mask, testing whether any lane is active, computing the successors' masks and, for each phi at the join, materializing the incoming values in a memory **slot** and combining them with a **blend** (lane-by-lane selection between two vectors under a mask, `vblendmps`/masked `vmovaps` in AVX-512).

Consider the line that loads a piece of matrix A into shared memory in the tiled multiplication (`samples/matmul.cu`):

```cpp
As[threadIdx.y][threadIdx.x] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
```

The condition depends on the thread (`a_col = t + threadIdx.x`), so the branch is **divergent** (varying). At run time almost all lanes satisfy the condition, but this cannot be proved statically. Linearization turned this line into:

```text
parent block: computes the condition; "then" mask = mask AND cond
then block:   any lane active? reads A[...] under the mask; stores to the slot
join block:   reads the slot, blends with 0.0 in the lanes where cond is false
              writes to As
```

Three blocks, mask operations, a slot with a blend and `any_lane` tests, for a single load and a `select` with zero. In an experiment with the variant without barrier (to isolate the effect), this structure alone cost about **265 ms** in the tiled matmul.

## The idea

**If-conversion** is the classic technique (Allen et al., 1983; used in predicated architectures such as IA-64) that trades control dependence for data dependence: both sides are executed unconditionally and a `select` chooses the result. In SSA, this means that the phis at the join become `select`s with the branch condition. In vectors, the `select` is lane by lane in a single instruction (`vblendmps` or a `mov` masked by a `k` register).

The restriction is that executing one side of the `if` in lanes whose condition is false must be **speculatively safe**:

- a **load** in a lane outside the matrix may touch an unmapped page and raise SIGSEGV;
- an **integer division** by zero raises `#DE` (SIGFPE), and the `if` may exist precisely to avoid it;
- a **store** changes observable memory, which cannot be speculated.

The solution: only convert sides without stores or operations that may raise an exception, and turn the loads into **guarded loads**, which only access memory in the lanes where the condition holds. AVX-512 supports this directly: masked loads (`vmovups` with `{k}{z}`, `vgatherdps` with a mask) suppress page faults in the disabled lanes.

## How it works, step by step

With a 4-lane warp and `K = 10`, `t = 8`, `threadIdx.x = 0, 1, 2, 3`:

```text
lane:                     0      1      2      3
a_col = t + threadIdx.x:  8      9     10     11
cond (a_col < K):         T      T      F      F
```

After if-conversion, the code looks like this (pseudocode):

```text
cond   = a_col < K                                   (vector of 4 booleans)
v      = __cpusim_cond_load(&A[row*K + a_col], cond, 0.0)
         -> reads A only in lanes 0 and 1; lanes 2 and 3 receive 0.0 and do not touch memory
x      = select(cond, v, 0.0)                        (the former phi at the join)
As[...] = x
```

No branch is left. Lanes 2 and 3, whose address `A[row*K + 10]` and `A[row*K + 11]` could be outside the array, never read anything.

### The steps

1. **Look for patterns.** For each conditional branch with a varying condition, the compiler checks whether it has one of three forms:
   - `if (c) { T }` followed by the join `J`;
   - `if (!c) { T }` (the side is on the "false" arm);
   - `if (c) { T } else { E }` followed by `J`.

   Each side must be a single basic block, have only the parent block as predecessor, go directly to the join, be in the same loop as the parent and contain no barrier. The join must have exactly those two predecessors and cannot be a loop header.

2. **Check that the side is safe** (`convertible`). The side must be **small** (up to 24 instructions) and contain only:
   - arithmetic and comparisons that raise no exception (additions, multiplications, floating-point operations, conversions, `select`, address computations);
   - calls to math functions that the compiler knows how to vectorize and reads of `threadIdx`/`blockIdx`;
   - simple loads (non-volatile, non-atomic).

   It cannot have **stores**, ordinary **calls**, **barriers**, **phis** or **integer divisions** (`/` and `%` on integers).

3. **Move the side into the parent** (`hoist_guarded`). The side's instructions are moved to the end of the parent block, before the branch. Each load becomes a call `__cpusim_cond_load.<T>(p, c, other)`, meaning "if `c`, read `*p`, otherwise return `other`".

4. **Replace the phis with selects.** In SSA, a **phi** at the start of the join defines a value chosen according to the incoming edge (`phi [v_t, %T], [v_o, %parent]`). Since the edge is now determined by the condition `c`, the phi is equivalent to `select(c, v_t, v_o)`, inserted in the parent after the moved instructions (which now dominate the point of use).

5. **Remove the branch.** The parent now jumps directly to the join, and the blocks of the sides are deleted.

6. **Repeat** the search from the beginning after each conversion, because the list of blocks changed, until there is nothing left to convert.

7. **Vectorize the guarded load.** In vector code generation, `__cpusim_cond_load` becomes a masked load with the mask `current_mask AND c`, going through the same [dynamic coalescing](06-dynamic-coalescing.md) as ordinary loads (so a contiguous access is still a single vector load).

## The code

The safety test for a side:

`src/compiler/simt.cpp:482-511`

```cpp
bool WarpBuilder::convertible(BasicBlock* bb) {
    unsigned n = 0;
    for (Instruction& inst : *bb) {
        if (inst.isTerminator()) break;
        if (++n > 24 || isa<PHINode>(inst)) return false;
        if (auto* ld = dyn_cast<LoadInst>(&inst)) {
            if (!ld->isSimple()) return false;
            continue;
        }
        if (auto* call = dyn_cast<CallInst>(&inst)) {
            if (special_reg(call) != SpecialReg::None || is_cond_load(call)) continue;
            Function* f = call->getCalledFunction();
            if (f && f->isIntrinsic() && isTriviallyVectorizable(f->getIntrinsicID())) continue;
            return false;
        }
        if (isa<BinaryOperator>(inst)) {
            switch (inst.getOpcode()) {
                case Instruction::UDiv: case Instruction::SDiv: case Instruction::URem: case Instruction::SRem:
                    return false;
                default:
                    continue;
            }
        }
        if (isa<CastInst>(inst) || isa<CmpInst>(inst) || isa<SelectInst>(inst) || isa<GetElementPtrInst>(inst) ||
            isa<UnaryOperator>(inst) || isa<FreezeInst>(inst))
            continue;
        return false;
    }
    return true;
}
```

It is an allow list: each accepted instruction kind has a `continue`; anything not on the list (store, unknown call, atomic) falls into the final `return false`. The excluded divisions are the **integer** ones (`UDiv`, `SDiv`, `URem`, `SRem`), which raise an exception when dividing by zero. Floating-point division is not on the exclusion list: in `float`, dividing by zero gives infinity or NaN, with no exception, so it is safe to execute in lanes that would not need it.

The transformation of loads into guarded loads:

`src/compiler/simt.cpp:515-542`

```cpp
void WarpBuilder::hoist_guarded(BasicBlock* from, BasicBlock* into, Value* guard) {
    Instruction* at = into->getTerminator();
    std::vector<Instruction*> insts;
    for (Instruction& inst : *from)
        if (!inst.isTerminator()) insts.push_back(&inst);
    for (Instruction* inst : insts) {
        auto* ld = dyn_cast<LoadInst>(inst);
        if (!ld) {
            inst->moveBefore(at);
            continue;
        }
        Type* t = ld->getType();
        std::string tn;
        raw_string_ostream os(tn);
        t->print(os);
        FunctionCallee fn = m_.getOrInsertFunction(
            kCondLoadPrefix + tn, FunctionType::get(t, {ld->getPointerOperandType(), Type::getInt1Ty(ctx_), t}, false));
        if (auto* f = dyn_cast<Function>(fn.getCallee())) {
            f->setDoesNotThrow();
            f->setOnlyReadsMemory();
            f->setWillReturn();
        }
        auto* call = CallInst::Create(fn, {ld->getPointerOperand(), guard, Constant::getNullValue(t)}, ld->getName(),
                                      at->getIterator());
        ld->replaceAllUsesWith(call);
        ld->eraseFromParent();
    }
}
```

- Instructions that are not loads are just moved before the parent's branch (`moveBefore(at)`).
- Each load becomes a call to `__cpusim_cond_load.<type>` (for example `__cpusim_cond_load.float`). The function is declared once per type with the attributes `nounwind`, `memory(read)` and `willreturn`, which lets the LLVM passes (GVN, LICM, DCE) treat it as a load: it can be eliminated if unused and moved as long as no intervening store invalidates it.
- The arguments are the pointer, the guard (`c` or `!c`) and the alternative value, which here is zero (`getNullValue`). The alternative value is irrelevant to the result, because the `select` that replaces the phi picks the value from the other side in the lanes where the guard is false.

The recognition of the patterns and the replacement of the phis with `select`:

`src/compiler/simt.cpp:568-605`

```cpp
            Value* c = br->getCondition();
            BasicBlock *t = nullptr, *e = nullptr, *j = nullptr;
            bool t_is_true = true;
            if (s0->getSingleSuccessor() && side(s0, s0->getSingleSuccessor()) && s0->getSingleSuccessor() == s1 &&
                joins_exactly(s1, &b, s0)) {
                t = s0, j = s1;  // if (c) { t } j
            } else if (s1->getSingleSuccessor() && side(s1, s1->getSingleSuccessor()) &&
                       s1->getSingleSuccessor() == s0 && joins_exactly(s0, &b, s1)) {
                t = s1, j = s0, t_is_true = false;  // if (!c) { t } j
            } else if (s0->getSingleSuccessor() && s0->getSingleSuccessor() == s1->getSingleSuccessor() &&
                       side(s0, s0->getSingleSuccessor()) && side(s1, s0->getSingleSuccessor()) &&
                       joins_exactly(s0->getSingleSuccessor(), s0, s1)) {
                t = s0, e = s1, j = s0->getSingleSuccessor();  // if (c) { t } else { e } j
            } else {
                continue;
            }

            IRBuilder<> ib(b.getTerminator());
            Value* not_c = ib.CreateNot(c, c->getName() + ".not");
            uni_->set_varying(not_c);
            hoist_guarded(t, &b, t_is_true ? c : not_c);
            if (e) hoist_guarded(e, &b, not_c);
            BasicBlock* other = e ? e : &b;  // where the "condition false" value comes from
            ib.SetInsertPoint(b.getTerminator());
            std::vector<PHINode*> phis;
            for (PHINode& phi : j->phis()) phis.push_back(&phi);
            for (PHINode* phi : phis) {
                Value* vt = phi->getIncomingValueForBlock(t);
                Value* vo = phi->getIncomingValueForBlock(other);
                Value* sel = t_is_true ? ib.CreateSelect(c, vt, vo, phi->getName()) : ib.CreateSelect(c, vo, vt, phi->getName());
                uni_->set_varying(sel);
                phi->replaceAllUsesWith(sel);
                phi->eraseFromParent();
            }
            b.getTerminator()->eraseFromParent();
            BranchInst::Create(j, &b);
            t->eraseFromParent();
            if (e) e->eraseFromParent();
```

- The three `if`s recognize the three forms described in step 1. `side` checks the conditions of a side (single predecessor, goes directly to the join, same loop, no barrier, `convertible`); `joins_exactly` checks the join.
- `hoist_guarded` is called with the right guard: `c` for the true side, `!c` for the false one.
- `other` is where the phi's "condition false" value comes from: the `else` block, if it exists, or the parent itself (in the case of the `if` without `else`, like the `? : 0.0f` of the example, where the phi receives `0.0` directly from the parent).
- Each phi becomes a `select` with the original condition, in the right order according to `t_is_true`.
- At the end, the parent's conditional branch is deleted and replaced by an unconditional jump to the join.

The vector generation of the guarded load:

`src/compiler/simt.cpp:1510-1527`

```cpp
Value* WarpBuilder::emit_cond_load(CallInst* call, Value* mask) {
    Value* ptr = call->getArgOperand(0);
    Value* c = call->getArgOperand(1);
    Value* other = call->getArgOperand(2);
    Type* t = call->getType();
    Value* m = uni_->is_varying(c) ? b_.CreateAnd(mask, vector(c)) : b_.CreateSelect(scalar(c), mask, no_lanes());
    if (!uni_->is_varying(call)) {
        // Everything uniform: one guarded scalar load.
        auto* v1 = FixedVectorType::get(t, 1);
        Value* go = b_.CreateAnd(scalar(c), any_lane(mask));
        Value* v = b_.CreateMaskedLoad(v1, scalar(ptr), Align(1), b_.CreateVectorSplat(1, go),
                                       b_.CreateVectorSplat(1, scalar(other)));
        return b_.CreateExtractElement(v, uint64_t(0));
    }
    const DataLayout& dl = m_.getDataLayout();
    Value* v = gather(t, ptr, dl.getABITypeAlign(t), m, call->getName());
    return b_.CreateSelect(m, v, vector(other));
}
```

- `m` is the effective mask: the active lanes **and** those where the guard holds (`mask AND c`).
- If everything is uniform (same address and same condition in all lanes), a single guarded scalar load is enough: a masked load of a single element, which only happens if the condition holds and some lane is active.
- Otherwise, it calls `gather` with the mask `m`. Despite the name, this function chooses the cheapest possible load (contiguous, broadcast or a real gather), as explained in [06-dynamic-coalescing.md](06-dynamic-coalescing.md). Lanes outside `m` do not touch memory.
- The final `select` puts the `other` value in the lanes outside `m`.

Finally, where if-conversion is called and why the profiling variant is left out:

`src/compiler/simt.cpp:2721-2728`

```cpp
    // The instrumented variant keeps the original branches so its per-thread
    // counts stay exact (if-conversion runs both sides' arithmetic).
    if (!profile_ && if_convert()) {
        dt_ = std::make_unique<DominatorTree>(f_);
        pdt_ = std::make_unique<PostDominatorTree>(f_);
        li_ = std::make_unique<LoopInfo>(*dt_);
        uni_ = std::make_unique<Uniformity>(f_, *dt_, *pdt_, *li_);
    }
```

The instrumented variant (`profile_`) counts how many instructions each thread executed. With if-conversion, the arithmetic of a side is also counted in the lanes that would not traverse it, and the counts would no longer reflect the thread's execution on the GPU. For this reason that variant keeps the original branches. After converting, the analyses (dominator and post-dominator trees, `LoopInfo`, uniformity) are recomputed, because the CFG changed and uniformity depends on the control dependences derived from the post-dominators.

## When it applies (and when it does not)

- **It applies** to **varying** (divergent) branches in the forms `if`, `if not` and `if/else`, with each side in a single block of up to 24 instructions, without stores, ordinary calls, barriers or integer divisions.
- **It does not apply** to **uniform** branches: those already become a scalar jump, and converting would execute one of the sides needlessly.
- **It does not apply** to sides that **write** to memory. For example, `if (i < n) y[i] = x[i];` remains linearized, because the side contains a store.
- **It does not apply** to large sides. The limit of 24 instructions exists because, with the conversion, the lanes that would not need the side also pay its cost. For a large side, this can end up more expensive than the branch.
- **It does not apply** when the branch or one of the sides contains a barrier, when a side is in a different loop from the parent, or when the join is a loop header.
- **It does not run in the profiling variant**, to keep the per-thread counts exact.
- **Side effect:** eliminating a varying `if` inside a loop can leave the loop with all branches uniform, and it is then generated as a [uniform loop](07-uniform-loops.md). This is what happens in the `t` loop of the tiled matmul, which contains the two lines `As[...] = ... ? ... : 0` and `Bs[...] = ... ? ... : 0`.

## Results

In the tiled matmul, measured on the variant without barrier (the experiment used to isolate the cost of the branch), the time dropped from **760 ms to 525 ms**, matching the naive matmul. There is no isolated measurement for other kernels.

## Limitations and trade-offs

- The lanes that would not need the side execute its arithmetic anyway. With the 24-instruction limit this is cheap, but it is not zero.
- Integer divisions prevent the conversion, even when the divisor is never zero. The compiler does not try to prove this.
- Only single-block sides are converted. An `if` whose side has more than one block (for example, with another `if` inside) remains linearized, even if the inner `if` is converted.
- The alternative value of the guarded load is always zero; the correct result depends on the `select` that replaces the phi. This is guaranteed by construction, but it means that the guarded load alone does not carry the full semantics of the `? :`.

## See also

- [00-basic-concepts.md](../00-basic-concepts.md): mask, active lane, divergence.
- [07-uniform-loops.md](07-uniform-loops.md): linearization and loops without masks.
- [06-dynamic-coalescing.md](06-dynamic-coalescing.md): how the guarded load is emitted.
- [09-state-at-barriers.md](09-state-at-barriers.md) and [10-barrier-pass.md](../iteration-2/10-barrier-pass.md): the context of the "without barrier" experiment.
- [22-what-did-not-work.md](../iteration-2/22-what-did-not-work.md): other attempts.
