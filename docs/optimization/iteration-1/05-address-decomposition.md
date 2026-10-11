# 05. Address decomposition into uniform base + varying offset

> **In one sentence:** when 32 threads read `B[k*N + col]`, the compiler splits the address into a part that is the same for all of them (`B + k*N`, a single number) and a part that changes from thread to thread but does not change inside the loop (`col`), and thereby takes almost all the address computation out of the loop.

## The problem

Each CUDA thread is a **lane** of an AVX-512 vector and a warp of 32 threads is processed per call (the general concepts are in [00-basic-concepts.md](../00-basic-concepts.md)). When the kernel accesses memory through a **varying** pointer (classified as non-uniform by the uniformity analysis), the vector code needs one address per lane.

The direct translation is to vectorize the whole address expression, obtaining **32 full 64-bit pointers** (a `<32 x ptr>`), and only then inspect them to choose the access form (whether the addresses are neighbors, whether they are all equal, etc.; that test is the subject of [06-dynamic-coalescing.md](06-dynamic-coalescing.md)).

Take the central loop of the naive matrix multiplication (`samples/matmul_naive.cu`):

```cpp
for (int k = 0; k < K; ++k) acc += A[row * K + k] * B[k * N + col];
```

In the direct translation, at each iteration:

1. we computed `k * N + col` for the 32 lanes (integer vector);
2. we converted to 64 bits and multiplied by 4 (the size of a `float`);
3. we added it to the pointer `B`, generating 32 64-bit pointers (4 zmm registers, since one zmm holds 8 qwords);
4. we tested whether the 32 pointers formed a simple pattern.

All of this **at every iteration**, even though the pattern (the lanes' pointers being consecutive) is loop-invariant. LICM cannot hoist any of it, because each pointer depends on `k`.

## The idea

The address of each lane is a linear expression over uniform values and varying values. The decomposition rewrites every per-thread address as:

```text
address of lane l  =  uniform base  +  offset[l]
                      (a single number,  (a vector of displacements
                       equal for all)     in bytes, one per lane)
```

- The **uniform base** contains everything that is the same for the 32 lanes: the pointer `B`, the loop counter `k`, `blockIdx`, the size `N`. It is a single scalar `ptr` value, which lives in a GPR.
- The **varying offset** contains only what depends on the thread. In CUDA, this is almost always something derived from `threadIdx`, which **does not change during the loop**.

The key point is the second one: since the offset depends only on `threadIdx`, it is **loop-invariant**. In SSA IR, the vector of offsets and every value computed only from it (such as the pattern tests of document 06) have operands defined outside the loop, and LLVM's LICM hoists them to the preheader. What remains inside the loop is the scalar base, which SCEV recognizes as an add recurrence (`{B + c, +, 4*N}`) and which IndVarSimplify/LSR reduce to a single `add` per iteration.

## How it works, step by step

Example: the read `B[k * N + col]`, with `col = blockIdx.x * 16 + threadIdx.x`, in a warp of 4 lanes with `threadIdx.x = 0, 1, 2, 3`. Assume `N = 100`, `blockIdx.x = 2`, `k = 5`.

The index of each lane is:

```text
lane:              0      1      2      3
k*N:             500    500    500    500     <- equal in all (uniform)
blockIdx.x*16:    32     32     32     32     <- uniform
threadIdx.x:       0      1      2      3     <- varying
index:           532    533    534    535
```

The compiler recursively walks the definition graph (use-def) of the index, separating the terms:

1. `k*N + col` is a sum. It decomposes each side: `k*N` is entirely uniform; `col` is another sum.
2. `col = blockIdx.x*16 + threadIdx.x`: `blockIdx.x*16` is uniform; `threadIdx.x` is varying.
3. It puts everything together: uniform part = `k*N + blockIdx.x*16 = 532`; varying part = `threadIdx.x = (0, 1, 2, 3)`.
4. It multiplies both parts by 4 (bytes per `float`) and adds the uniform part to the pointer `B`:

```text
base    = B + 4 * (k*N + blockIdx.x*16)      a single pointer
offsets = 4 * threadIdx.x = (0, 4, 8, 12)    does not change with k
```

On the next round (`k = 6`), only the `base` changes (it adds `4*N`). The `offsets` are the same, so the compiler computes them once before the loop.

### What kinds of expression it can split

The decomposition descends the use-def chain of the index and only crosses operations over which the separation `u + v` distributes:

| Operation | How it splits | Note |
|---|---|---|
| uniform value | everything goes to the base | |
| `a + b` | adds the bases and adds the offsets | depends on there being no overflow (see below) |
| `a or b` with `disjoint` | same as the sum | this is what LLVM makes of `t + threadIdx.x` when the bits do not overlap |
| `(u + v) * c`, `c` uniform | `u*c` goes to the base, `v*c` to the offsets | `(t + ty) * N = t*N + ty*N` |
| `(u + v) << c`, `c` uniform | same as multiplication | `x << c = x * 2^c` |
| extension from 32 to 64 bits | splits the inner value | remembering whether the extension is signed or unsigned |
| anything else | everything goes to the offsets | safe case, but with no gain |

### The care with integer overflow

Indices in CUDA are generally 32-bit `int`, but addresses are 64-bit; between the two there is a `sext` or `zext` in the IR. The identity `ext(a + b) = ext(a) + ext(b)` only holds if the sum `a + b` **does not wrap around** in 32 bits (in the signed sense for `sext`, unsigned for `zext`).

Example in 8-bit unsigned: `200 + 100` in 8 bits gives `44`; `zext` to 16 bits gives `44`. But `zext(200) + zext(100)` in 16 bits gives `300`.

For this reason the compiler consults the wrap flags of the IR instructions. In LLVM, `add nsw` (*no signed wrap*) and `add nuw` (*no unsigned wrap*) declare that, if the operation wraps in the corresponding sense, the result is `poison`; the optimizer can therefore assume it does not wrap. The front end emits `nsw` for signed `int` arithmetic because signed overflow is undefined behavior in C++.

- `nsw` allows distributing a `sext` over the sum;
- `nuw` allows distributing a `zext` over the sum.

There is an important special case. CUDA code mixes `int` with `threadIdx`, which is `unsigned`. By the usual C++ conversion rules, `row * K + threadIdx.x` is an `unsigned` sum, emitted **with neither `nsw` nor `nuw`** (unsigned arithmetic has defined modular semantics). With the strict rule, almost no real index would be split. The adopted rule is: **under signed extension, the unmarked sum is also split**. The justification: the result would only differ from the exact one if the signed sum left the range of `int` (about ±2^31). In that case the original index would already be pointing about 2^31 elements (billions) away from the intended position, outside any allocation, that is, the program would already be reading invalid memory. Under **unsigned** extension this is not done, because there wrapping around can be legitimate program behavior.

## The code

The structure that represents a decomposed address:

`src/compiler/simt.cpp:102-106`

```cpp
    struct SplitAddress {
        Value* base;     // uniform pointer (scalar)
        Value* offsets;  // varying byte offsets, <32 x i64>
        uint64_t scale = 1;  // a per-thread array: a thread's byte offsets count W times (lane-major layout)
    };
```

`base` is the scalar pointer; `offsets` is the vector of displacements in bytes (one 64-bit integer per lane). `scale` is explained further below.

The decomposition of an integer index:

`src/compiler/simt.cpp:900-949`

```cpp
std::pair<Value*, Value*> WarpBuilder::split_index(Value* idx, bool is_signed) {
    auto* i64 = b_.getInt64Ty();
    auto* v64 = FixedVectorType::get(i64, W);
    if (!uni_->is_varying(idx)) {
        Value* s = scalar(idx);
        return {is_signed ? b_.CreateSExtOrTrunc(s, i64) : b_.CreateZExtOrTrunc(s, i64), Constant::getNullValue(v64)};
    }
    if (auto* ext = dyn_cast<CastInst>(idx); ext && (isa<SExtInst>(ext) || isa<ZExtInst>(ext)))
        return split_index(ext->getOperand(0), isa<SExtInst>(ext));
    if (auto* add = dyn_cast<BinaryOperator>(idx);
        add && (add->getOpcode() == Instruction::Add ||
                (add->getOpcode() == Instruction::Or && cast<PossiblyDisjointInst>(add)->isDisjoint()))) {
        // ...
        bool disjoint = add->getOpcode() == Instruction::Or;
        // ...
        bool no_wrap = disjoint || idx->getType()->isIntegerTy(64) || is_signed || add->hasNoUnsignedWrap();
        if (no_wrap) {
            auto [u0, v0] = split_index(add->getOperand(0), is_signed);
            auto [u1, v1] = split_index(add->getOperand(1), is_signed);
            return {b_.CreateAdd(u0, u1), b_.CreateAdd(v0, v1)};
        }
    }
    // ...
    if (auto* bo = dyn_cast<BinaryOperator>(idx);
        bo && (bo->getOpcode() == Instruction::Mul || bo->getOpcode() == Instruction::Shl) &&
        (is_signed || bo->hasNoUnsignedWrap() || idx->getType()->isIntegerTy(64))) {
        Value* a = bo->getOperand(0);
        Value* c = bo->getOperand(1);
        if (bo->getOpcode() == Instruction::Mul && uni_->is_varying(c) && !uni_->is_varying(a)) std::swap(a, c);
        if (!uni_->is_varying(c) && uni_->is_varying(a)) {
            auto [u, v] = split_index(a, is_signed);
            Value* cs = scalar(c);
            Value* c64 = is_signed && bo->getOpcode() == Instruction::Mul ? b_.CreateSExtOrTrunc(cs, i64)
                                                                          : b_.CreateZExtOrTrunc(cs, i64);
            if (bo->getOpcode() == Instruction::Mul)
                return {b_.CreateMul(u, c64), b_.CreateMul(v, b_.CreateVectorSplat(W, c64))};
            return {b_.CreateShl(u, c64), b_.CreateShl(v, b_.CreateVectorSplat(W, c64))};
        }
    }
    Value* v = vector(idx);
    v = is_signed ? b_.CreateSExtOrTrunc(v, v64) : b_.CreateZExtOrTrunc(v, v64);
    return {ConstantInt::get(i64, 0), v};
}
```

The function returns a pair: (scalar uniform part, vector varying part), whose sum is the original index in 64 bits. Reading from top to bottom:

- **Uniform value** (`!is_varying`): goes entirely to the uniform part; the varying part is a vector of zeros.
- **Extension** (`sext`/`zext`): descends to the inner value, remembering in `is_signed` whether the extension was signed. This memory is what decides the overflow rules.
- **Sum or `or disjoint`**: the variable `no_wrap` summarizes when it is safe to split. It is safe if it is an `or disjoint`, if the computation is already 64-bit (there is no extension to cross), if we are under signed extension (the special case explained above) or if the sum has the `nuw` mark. Then it splits both sides and adds the corresponding parts.
- **Multiplication or shift by a uniform value**: if one side is varying and the other uniform (the `swap` always puts the uniform one in `c`), it splits the varying side and multiplies both parts by `c`. This is the rule `(t + ty) * N = t*N + ty*N`.
- **Anything else**: gives up splitting and puts the whole value in the varying part. It remains correct, it just gains nothing.

The decomposition of the pointer, which traverses the GEPs:

`src/compiler/simt.cpp:955-988`

```cpp
WarpBuilder::SplitAddress WarpBuilder::split_address(Value* ptr) {
    auto* i64 = b_.getInt64Ty();
    auto* v64 = FixedVectorType::get(i64, W);
    if (auto* a = dyn_cast<AllocaInst>(ptr); a && private_elem_.count(a)) {
        // Lane l's copy of the array starts at base + l * elem.
        SmallVector<Constant*, 64> lanes;
        for (unsigned l = 0; l < W; ++l) lanes.push_back(ConstantInt::get(i64, l * private_elem_[a]));
        return {private_base(a), ConstantVector::get(lanes), W};
    }
    if (!uni_->is_varying(ptr)) return {scalar(ptr), Constant::getNullValue(v64)};
    if (auto* gep = dyn_cast<GetElementPtrInst>(ptr)) {
        const DataLayout& dl = m_.getDataLayout();
        SplitAddress inner = split_address(gep->getPointerOperand());
        Value* uniform = ConstantInt::get(i64, 0);
        Value* varying = inner.offsets;
        for (gep_type_iterator it = gep_type_begin(gep), end = gep_type_end(gep); it != end; ++it) {
            if (StructType* st = it.getStructTypeOrNull()) {
                // ...
                continue;
            }
            uint64_t size = it.getSequentialElementStride(dl) * inner.scale;
            auto [u, v] = split_index(it.getOperand(), true);
            uniform = b_.CreateAdd(uniform, b_.CreateMul(u, ConstantInt::get(i64, size)));
            varying = b_.CreateAdd(varying, b_.CreateMul(v, ConstantVector::getSplat(ElementCount::getFixed(W),
                                                                                    ConstantInt::get(i64, size))));
        }
        return {b_.CreateGEP(b_.getInt8Ty(), inner.base, uniform), varying, inner.scale};
    }
    // Opaque varying pointer: offsets from address zero.
    Value* addrs = b_.CreatePtrToInt(vector(ptr), v64);
    return {ConstantPointerNull::get(cast<PointerType>(ptr->getType()->getScalarType())), addrs};
}
```

A **GEP** (`getelementptr`) is the LLVM instruction that does only address arithmetic, without accessing memory: given a base pointer and a list of indices, it adds `index * stride` for each level of the aggregate type (the stride of an array is the allocated size of the element; an index into a struct is constant and adds the field offset, given by the `DataLayout`). The indices are sign-extended to the pointer width, which is why the function calls `split_index(..., true)`. Multidimensional accesses such as `As[ty][tx]` appear as a GEP with several indices or as chained GEPs. The function:

- if the pointer is uniform, returns it as the base, with zero offsets;
- if it is a GEP, first decomposes the inner pointer (recursion: there can be chained GEPs, as in `As[ty][tx]`), then each index with `split_index`, and multiplies each part by the element size in bytes. The uniform part becomes a new scalar GEP over the base; the varying part accumulates in the offsets;
- if it is a varying pointer with no recognizable structure (for example, loaded from memory, or the result of a `phi`/`select` of pointers), uses the null base and the 32 full addresses (`ptrtoint`) as offsets. This is the old, more expensive case, but always correct.

### The extension for local arrays (`scale`)

The first `if` handles a case that came later: arrays declared inside the kernel, such as `float v[8]`, which remain as `alloca` after SROA/mem2reg (typically because they are indexed with a dynamic index). Each thread has its own copy. They are stored **interleaved by lane** (SoA layout): element `j` of all lanes is contiguous in memory. Thus, when all threads read `v[j]` with the same `j`, the 32 values occupy 128 consecutive bytes and the read becomes two zmm `vmovups` instead of a gather.

In this layout, advancing one element in a thread's array means skipping `W` elements in memory (one for each lane). That is what `scale = W` records: the sizes of the following indices are multiplied by `W`, and the initial offsets are `l * element_size` for lane `l`.

## When it applies (and when it does not)

- **It always runs** for loads and stores through a varying pointer, in global memory as well as in shared memory (`__shared__`) and in local arrays. There is no on/off switch.
- **It gains most** when the varying part depends only on `threadIdx` and on constants, and the uniform part carries the loop counter. This is the dominant pattern in CUDA (`A[row*K + k]`, `B[k*N + col]`, `As[ty][k]`).
- **It gains little or nothing** when the index depends on a value read from memory (for example, `x[col_idx[j]]` in a sparse matrix): the varying part changes at every iteration and cannot be hoisted. The split adds no cost in that case.
- **It does not split** unmarked sums under unsigned extension (for example, `size_t` indices computed with `unsigned`), because there the overflow may be intentional. In those cases the whole index stays in the varying part. The static analysis of document 12 has an extra test that sometimes proves the absence of overflow from value ranges (see [13-no-overflow-proof.md](../iteration-2/13-no-overflow-proof.md)).
- **Assumption made:** under signed extension, an `int` sum that overflows is treated as if it did not overflow. Only a program that already accesses memory outside any allocation could notice the difference.

## Results

There was no isolated measurement of this optimization. It is the foundation of the other two that came next: the pattern tests of the [dynamic coalescing](06-dynamic-coalescing.md) only leave the loop because they look at the offsets, which are invariant. Without the decomposition, those tests would be redone at every iteration over 32 full pointers. The numbers in document 06 (for example, the naive matmul dropping from 2340 to 1265 ms with one of the refinements) presuppose this decomposition.

Qualitatively, what changes inside the loop is: instead of computing four registers of 64-bit pointers and testing them, what remains is a scalar addition on the base and the load itself.

## Limitations and trade-offs

- The decomposition only recognizes sums, `or disjoint`, multiplications and shifts by a uniform value, and extensions. A subtraction or a division in the middle of the index makes the whole subtree fall into the varying part.
- The rule for unmarked sums under `sext` is an assumption about well-behaved programs, not a proof. It was chosen because the cost of not making it would be losing the optimization in practically all real CUDA code.
- The offset is stored in 64 bits per lane. For 32 lanes that is four zmm registers; this is why document 06 avoids keeping them live on the common path.

## See also

- [00-basic-concepts.md](../00-basic-concepts.md): lane, warp, uniform and varying.
- [06-dynamic-coalescing.md](06-dynamic-coalescing.md): what is done with the base and the offsets.
- [12-static-access-analysis.md](../iteration-2/12-static-access-analysis.md): when the pattern of the offsets is proven at compile time.
- [13-no-overflow-proof.md](../iteration-2/13-no-overflow-proof.md): proving the absence of overflow instead of assuming it.
- [09-state-at-barriers.md](09-state-at-barriers.md): where local arrays live when the warp suspends.
