# 12. Static access analysis

> **In one sentence:** since the JIT already knows, at compile time, the `threadIdx` of each lane, the compiler can *prove* whether each 16-lane chunk reads or writes neighboring addresses (contiguous), a single address (broadcast) or scattered addresses (gather), and then directly generates the right instruction, with no run-time tests.

## The problem

The starting point was the "tiled" matmul (which copies tiles of the matrices into the block's *shared memory*; see [00-basic-concepts.md](../00-basic-concepts.md)), slower than it should be. The user's first hypothesis was that the shared memory was "far" from the core, and the proposal was to allocate it with `mmap` with special flags and emit *prefetch* to "keep it close to the core".

The measurement showed something else:

- The shared memory was already in the L1D. There was nothing to bring closer.
- The compute loop of the 256-lane variant (see [02-width-variants.md](../iteration-1/02-width-variants.md)) was already ideal: for each step, 1 load of `Bs`, 16 `vmulps` with broadcast and 16 `vaddps`, keeping 16 accumulators in registers. This loop took about **36%** of the cycles.
- The other **~64%** were in the global-to-shared copy (the part in which each thread copies one element of `A` into `As` and one of `B` into `Bs`). Within those 64%:

| Where the cycles went | Share of the total |
|---|---|
| *Spills* (saves and reloads of zmm on the stack) | 23% |
| Masks (active lanes and results of the pattern tests) | 14% |
| Address arithmetic | 11% |
| Others | 8% |
| Actual loads and stores (global + shared) | 7% |

That is: the real memory accesses were only ~7% of the cycles outside the compute loop. The rest was overhead of classifying, at run time, the pattern of each access and of keeping alive the data needed for that classification.

The cause was in how the compiler handled an access with a different address per lane. It built a vector of 64-bit offsets (one per lane) and, **at run time**, tested each 16-lane chunk: "are the addresses consecutive? are they all equal? neither?". Each answer had a different code path, and LLVM, when optimizing, ended up making 4 copies of the loop, one for each combination of results. With 256 lanes, the vector of 64-bit offsets is 256 x 8 = 2 KiB, that is, 32 zmm registers: the entire architectural register file of AVX-512. It cannot stay in registers together with the accumulators and the rest of the state, and the register allocator splits it between registers and stack: hence the spills.

The 256-lane kernel of the tiled matmul had **8108 instructions** and several tests of this kind.

## The idea

The run-time classification exists because, in general, the compiler does not know the relation between the lanes' addresses. But, if each lane's offset can be written as a function whose dependence on the lane is known at compile time, the classification can be done *statically*, as a proof, and the generated code needs no test at all.

This information comes from the specialized JIT ([01-specialized-jit.md](../iteration-1/01-specialized-jit.md)): the kernel is compiled already knowing the block shape (for example 16x16). With this, the `threadIdx.x` and `threadIdx.y` of each lane are compile-time **constants**. What is still unknown (the `blockIdx`, the matrix width `N`, the loop counter `t`) is **uniform**: equal for all the lanes of the warp.

So each lane's offset can be written as a sum of terms of the form

```text
offset(lane) = c1 * k1[lane] + c2 * k2[lane] + ... + constant[lane]
```

where each `c` is a uniform value (unknown at compile time, but equal in all lanes) and each `k[lane]` is an integer known at compile time. This is the **linear form** (`LinearForm`): a linear combination of unknown scalar coefficients, in which everything that varies per lane is in the constant vectors. It is analogous to an SCEV affine expression, but with a "step" given by an arbitrary vector of per-lane constants, not by a single loop step.

Within a 16-lane chunk:

- if, in every term with an unknown coefficient, `k[lane]` is the same across the 16 lanes, that term adds the same value `c * k` to all of them: it is a translation of the whole chunk and does not change the differences between lanes;
- the differences between lanes come to depend only on the constant part; if it grows by `size` bytes per lane (4 for `float`), the chunk is **contiguous, proven**; if it is equal across the 16 lanes, it is a **broadcast, proven**.

The value of the coefficients is never needed for the proof; only for the address. And the chunk's address stops being a vector of 16 64-bit integers: it becomes **a single scalar** (the offset of the first lane), computed with scalar arithmetic over the coefficients.

## How it works, step by step

Reduced example: a 4x4 block (16 threads) and chunks of 4 lanes (in the real code the chunks have 16 lanes; the logic is the same). The thread of lane `l` has `tx = l % 4` and `ty = l / 4`:

```text
lane: 0  1  2  3 | 4  5  6  7 | 8  9 10 11 | 12 13 14 15
tx:   0  1  2  3 | 0  1  2  3 | 0  1  2  3 |  0  1  2  3
ty:   0  0  0  0 | 1  1  1  1 | 2  2  2  2 |  3  3  3  3
      chunk 0      chunk 1      chunk 2       chunk 3
```

The copy of `A` into shared in the tiled matmul is something like:

```cpp
row = blockIdx.y * 4 + threadIdx.y;
As[ty][tx] = A[row * N + t * 4 + tx];
```

In bytes, the offset is:

```text
4 * (row * N + t*4 + tx)
= 4 * ((by*4 + ty) * N + t*4 + tx)
= (by*N) * 16        <- coefficient by*N, number 16 in all lanes
+ N * (4*ty)         <- coefficient N, number 4*ty per lane
+ t * 16             <- coefficient t, number 16 in all lanes
+ 4*tx               <- constant part
```

Writing each term as "coefficient: numbers per lane":

| Term | Chunk 0 | Chunk 1 | Chunk 2 | Chunk 3 |
|---|---|---|---|---|
| `by*N` | 16 16 16 16 | 16 16 16 16 | 16 16 16 16 | 16 16 16 16 |
| `N` | 0 0 0 0 | 4 4 4 4 | 8 8 8 8 | 12 12 12 12 |
| `t` | 16 16 16 16 | 16 16 16 16 | 16 16 16 16 | 16 16 16 16 |
| constant | 0 4 8 12 | 0 4 8 12 | 0 4 8 12 | 0 4 8 12 |

Now the check, chunk by chunk:

1. Do the terms with an unknown coefficient (`by*N`, `N`, `t`) have the same number in the 4 lanes of each chunk? Yes: within a chunk, `ty` does not change. So they only shift the whole chunk.
2. The constant part, measured from the chunk's first lane, is `0, 4, 8, 12`, that is, `i * 4`. Contiguous.
3. The address of chunk 2, for example, is `by*N*16 + N*8 + t*16 + 0`: a scalar computation, done once.

Result: each chunk becomes a single masked load (`llvm.masked.load`, which on AVX-512 is a `vmovups` with a mask register `k`) from a scalar address. There is no test, no vector of 64-bit offsets and no duplicated versions of the loop.

Compare with a "transposed" access, `A[tx * N + ty]`. The term with coefficient `N` would have numbers `0, 4, 8, 12` within a single chunk. Since `N` is unknown, the distance between neighboring lanes is `4*N` bytes, with no known value: neither pattern can be proven (with `N = 1` it would be contiguous, with any other value it would not), and the compiler falls back to the old path, with run-time tests. This is the correct behavior: the proof is conservative.

### The case "all chunks read the same row"

In the read `Bs[k][threadIdx.x]` inside the compute loop, with a block 16 wide, the offset is `k*64 + 4*tx`. It does not depend on `ty`. So **all** the chunks read exactly the same 16 elements. Instead of doing one load per chunk (16 identical loads for 256 lanes), the compiler does **one** load and replicates the result. This case was added because, without it, the naive matmul had regressed (see Results).

### Extensions needed for the proof to work on real code

The IR that comes from Clang does not have the clean shape of the example. Three constructs needed handling:

- **Width extensions** (`zext` and `sext` from `i32` to `i64`). Distributing the extension over a sum, `ext(a + b) = ext(a) + ext(b)`, is only valid if the sum does not wrap around in 32 bits. This holds when the operation has the `nuw` flag (for `zext`) or `nsw` (for `sext`), whose semantics in LLVM is "if there is a wrap, the result is poison", or when it is an `or disjoint` (an `or` whose operands have no 1 bits in common, and which is therefore equal to the sum, with no carry). Under `sext`, an unflagged sum is also accepted, with the same justification already used in `split_index` (see [05-address-decomposition.md](../iteration-1/05-address-decomposition.md)): if an `int` index overflowed, it would point some 2^31 elements away, outside any allocation. The unflagged case under `zext` is handled in document [13-no-overflow-proof.md](13-no-overflow-proof.md).
- **Product of two uniforms.** When the block has more threads than the variant has lanes (for example a 16x16 block in the 32-lane variant), the `threadIdx.y` of a group of lanes is not constant: it is "a uniform (the row where the group starts) plus a per-lane constant". Multiplied by the width `N` (also uniform), each term becomes "uniform times uniform times constant". The linear form remains closed under this operation: the new coefficient is the product (scalar, emitted as a plain `mul`) of the two uniforms.
- **`shl` by a constant** `c` is treated as multiplication by `2^c`.

### Rematerialization of cheap values

Along with the analysis came a related change: `remat_ok`. In a kernel with `__syncthreads`, values live across the barrier are kept in a frame in memory (see [09-state-at-barriers.md](../iteration-1/09-state-at-barriers.md)). An expression such as `t + threadIdx.y` (with `t` the tile loop counter) was kept as a 256-lane vector of `i32`: 1 KiB per tile, written and reread. But it can be recomputed from the scalar `t` with a broadcast and an add. With `remat_ok`, the compiler clones this computation into the block that uses it, instead of keeping it in the frame. Besides saving memory traffic, this keeps the expression analyzable: a vector read from the frame is opaque to `linear_form`, whereas the cloned computation can be decomposed into a linear form.

## The code

### The structure of the linear form

`src/compiler/simt.cpp:117-127`

```cpp
    // Varying byte offsets as a sum of terms coef * lanes[i] (coef: uniform
    // scalar, nullptr for 1; lanes: constants). Known when the thread index of
    // every lane is a compile-time constant (block shape fixed by the JIT).
    struct LinearForm {
        std::vector<std::pair<Value*, std::vector<int64_t>>> terms;
    };
    // ext: the value is seen through a zero (1) or sign (2) extension to i64.
    bool linear_form(Value* v, LinearForm& out, int ext = 0);
    // Per 16-lane chunk, from a linear form: 'c' contiguous, 'b' one address,
    // 0 unknown at compile time; and the chunk's first offset (scalar).
    std::vector<char> static_chunks(const LinearForm& f, uint64_t size, std::vector<Value*>& firsts);
```

Each term is a pair: the coefficient (a scalar value of the program, or `nullptr` when the coefficient is simply 1, that is, the constant part) and a vector with one integer per lane. `linear_form` tries to write a value in this format; `static_chunks` looks at the result and classifies each 16-lane chunk as `'c'` (contiguous), `'b'` (broadcast) or `0` (unknown).

### Building the linear form: leaves and conversions

`src/compiler/simt.cpp:1000-1018`

```cpp
bool WarpBuilder::linear_form(Value* v, LinearForm& out, int ext) {
    out.terms.clear();
    if (auto* c = dyn_cast<Constant>(v)) {
        std::vector<int64_t> lanes(W);
        for (unsigned i = 0; i < W; ++i) {
            auto* e = dyn_cast_or_null<ConstantInt>(c->getAggregateElement(i));
            if (!e) return false;
            lanes[i] = ext == 1 ? int64_t(e->getZExtValue()) : e->getSExtValue();
        }
        out.terms.push_back({nullptr, std::move(lanes)});
        return true;
    }
    if (Value* s = getSplatValue(v)) {
        if (ext) s = ext == 1 ? b_.CreateZExt(s, b_.getInt64Ty()) : b_.CreateSExt(s, b_.getInt64Ty());
        out.terms.push_back({s, std::vector<int64_t>(W, 1)});
        return true;
    }
    if (auto* cast = dyn_cast<CastInst>(v); cast && !ext && (isa<ZExtInst>(cast) || isa<SExtInst>(cast)))
        return linear_form(cast->getOperand(0), out, isa<ZExtInst>(cast) ? 1 : 2);
```

These are the base cases of the recursion. A constant vector (for example the `threadIdx.x` of each lane, known thanks to the JIT) becomes a term with no coefficient. A *splat* (a vector with the same scalar in all lanes, recognized by `getSplatValue`; it is the representation of a uniform) becomes a term with that coefficient and the number 1 in all lanes. A `zext`/`sext` extension is traversed, recording in `ext` which one it was, because the operations below it will need to justify that they do not wrap around in 32 bits; the leaves below it are extended with the same operation (`CreateZExt`/`CreateSExt` on the coefficient, `getZExtValue`/`getSExtValue` on the constants).

### Additions, subtractions and the overflow rules

`src/compiler/simt.cpp:1021-1042`

```cpp
    // Under an extension, the operation must not wrap in the narrow type for
    // the extension to distribute over it. Under a sign extension the same
    // assumption as split_index: an int index that wraps would point ~2^31
    // elements away, outside any allocation (CUDA indexes mix int and the
    // unsigned threadIdx, so their adds carry no flags).
    bool disjoint = bo->getOpcode() == Instruction::Or && cast<PossiblyDisjointInst>(bo)->isDisjoint();
    if (ext == 1 && !disjoint) {
        auto* ob = dyn_cast<OverflowingBinaryOperator>(bo);
        if (!ob) return false;
        if (!ob->hasNoUnsignedWrap()) {
            // No flag, but the value ranges may still rule out wrapping
            // ...
            if (r != OverflowResult::NeverOverflows) return false;
        }
    }
```

Under a `zext`, an operation is only accepted if it provably does not wrap around in 32 bits (the `nuw` flag, or `or disjoint`, or the value-range proof explained in [document 13](13-no-overflow-proof.md)). Under a `sext`, nothing is required, by the justification in the comment. After that, `Add`, `Sub` and `or disjoint` simply merge the term lists of the two sides (flipping the sign of the numbers, in the case of subtraction).

### Multiplications: by a constant and by a uniform

`src/compiler/simt.cpp:1067-1090`

```cpp
            } else if (!constant(b)) {
                std::swap(a, b);
                if (!constant(b)) {
                    // a * splat(s): every coefficient times s (a scalar product).
                    auto splat_of = [](const LinearForm& f) -> Value* {
                        if (f.terms.size() != 1 || !f.terms[0].first) return nullptr;
                        for (int64_t x : f.terms[0].second)
                            if (x != 1) return nullptr;
                        return f.terms[0].first;
                    };
                    Value* s = splat_of(b);
                    if (!s) std::swap(a, b), s = splat_of(b);
                    if (!s) return false;
                    for (auto& t : a.terms) t.first = t.first ? b_.CreateMul(t.first, s) : s;
                    out.terms = std::move(a.terms);
                    return true;
                }
            }
            // a * constant lanes, term by term (splat(s) * c is the term s * c).
            const std::vector<int64_t>& k = b.terms[0].second;
            for (auto& t : a.terms)
                for (unsigned i = 0; i < W; ++i) t.second[i] *= k[i];
            out.terms = std::move(a.terms);
            return true;
```

A multiplication only preserves the linear form if one of the sides is constant per lane or a pure uniform. If one side is constant, each per-lane number of the other side is multiplied by it (`4 * ty` in the example). If one side is a pure uniform (`splat(s)`), each coefficient of the other side is multiplied by `s`: this is the "product of two uniforms" case (`(group_row + ty) * N`). Multiplying two values that vary per lane in an unknown way is not linear and the function gives up.

### Classifying each chunk

`src/compiler/simt.cpp:1097-1126`

```cpp
std::vector<char> WarpBuilder::static_chunks(const LinearForm& f, uint64_t size, std::vector<Value*>& firsts) {
    const unsigned chunks = W / 16;
    std::vector<char> kind(chunks, 0);
    firsts.assign(chunks, nullptr);
    auto* i64 = b_.getInt64Ty();
    for (unsigned h = 0; h < chunks; ++h) {
        const unsigned l0 = h * 16;
        bool uniform_terms = true;  // the uniform coefficients add the same to every lane of the chunk
        std::vector<int64_t> delta(16, 0);
        Value* first = ConstantInt::get(i64, 0);
        int64_t first_const = 0;
        for (const auto& [coef, lanes] : f.terms) {
            for (unsigned i = 0; i < 16; ++i) {
                if (!coef) delta[i] += lanes[l0 + i] - lanes[l0];
                else if (lanes[l0 + i] != lanes[l0]) uniform_terms = false;
            }
            if (!coef) first_const += lanes[l0];
            else if (lanes[l0]) first = b_.CreateAdd(first, b_.CreateMul(coef, ConstantInt::get(i64, lanes[l0])));
        }
        if (!uniform_terms) continue;
        firsts[h] = b_.CreateAdd(first, ConstantInt::get(i64, first_const));
        bool contig = true, same = true;
        for (unsigned i = 0; i < 16; ++i) {
            contig &= delta[i] == int64_t(i * size);
            same &= delta[i] == 0;
        }
        kind[h] = contig ? 'c' : same ? 'b' : 0;
    }
    return kind;
}
```

This is the table of the 4x4 example, generalized. For each chunk `h` (lanes `l0` to `l0 + 15`): the terms with a coefficient must have the same number in the 16 lanes (`uniform_terms`); the constant part is accumulated as a difference relative to the first lane (`delta`). If `delta[i] == i * size` for all `i`, it is contiguous; if all are zero, it is a broadcast. At the same time `firsts[h]` is built, the scalar computation of the address of the chunk's first lane.

### The static path at the start of `gather`

`src/compiler/simt.cpp:1266-1302`

```cpp
    // Pattern known at compile time (constant thread indices): straight-line
    // code, no tests, no vector of 64-bit offsets.
    if (LinearForm form; linear_form(addr.offsets, form)) {
        std::vector<Value*> firsts;
        std::vector<char> kind = static_chunks(form, size, firsts);
        if (std::all_of(kind.begin(), kind.end(), [](char k) { return k != 0; })) {
            // Every chunk reads the same 16 elements (Bs[k][threadIdx.x] in a
            // 16-wide block): one load, replicated.
            bool one_row = chunks > 1 && std::all_of(kind.begin(), kind.end(), [](char k) { return k == 'c'; });
            for (const auto& t : form.terms)
                for (unsigned h = 1; h < chunks && one_row; ++h) one_row = t.second[h * 16] == t.second[0];
            if (one_row) {
                Value* any_mask = half(mask, 0);
                for (unsigned h = 1; h < chunks; ++h) any_mask = b_.CreateOr(any_mask, half(mask, h));
                Value* row = b_.CreateMaskedLoad(v16, b_.CreateGEP(b_.getInt8Ty(), addr.base, firsts[0]), align, any_mask,
                                                 PoisonValue::get(v16));
                SmallVector<int, 64> rep;
                for (int i = 0; i < int(W); ++i) rep.push_back(i % 16);
                return b_.CreateShuffleVector(row, rep, name);
            }
            std::vector<Value*> parts(chunks);
            for (unsigned h = 0; h < chunks; ++h) {
                Value* m = half(mask, h);
                Value* base = b_.CreateGEP(b_.getInt8Ty(), addr.base, firsts[h]);
                if (kind[h] == 'c') {
                    parts[h] = b_.CreateMaskedLoad(v16, base, align, m, PoisonValue::get(v16));
                } else {
                    // ...
                    Value* any = b_.CreateICmpNE(b_.CreateBitCast(m, i16), b_.getInt16(0));
                    Value* p = b_.CreateSelect(any, base, b_.CreatePointerCast(ctx_arg_, base->getType()));
                    parts[h] = b_.CreateVectorSplat(16, b_.CreateAlignedLoad(elem, p, Align(1)));
                }
            }
            return concat_chunks(parts);
        }
    }
```

If every chunk was classified (none is `0`), the load is emitted in a straight line, with no branch:

- **Single row:** if all the chunks are contiguous and each term has, in the first lane of each chunk, the same number as in lane 0, then all the chunks start at the same address. One load is done with the union (`or`) of the chunks' masks (`any_mask`), and the result is replicated across all the chunks with a `shufflevector` with mask `i % 16`.
- **Contiguous:** a masked load from `firsts[h]`. The mask guarantees that inactive lanes do not read memory.
- **Broadcast:** a scalar load replicated across the 16 lanes. A scalar load has no mask, so, if the chunk has no active lane, the computed address may be invalid and the load could cause a page fault. In that case the `select` swaps the address for that of the warp context (`ctx_arg_`), which is always readable, and the value read is discarded.

If some chunk was left `0`, the code falls back to the old path, with run-time tests (see [06-dynamic-coalescing.md](../iteration-1/06-dynamic-coalescing.md)).

### The static path at the start of `scatter`

`src/compiler/simt.cpp:1446-1455`

```cpp
    if (LinearForm form; linear_form(addr.offsets, form)) {
        std::vector<Value*> firsts;
        std::vector<char> kind = static_chunks(form, size, firsts);
        if (std::all_of(kind.begin(), kind.end(), [](char k) { return k == 'c'; })) {
            for (unsigned h = 0; h < chunks; ++h)
                b_.CreateMaskedStore(half(vals, h), b_.CreateGEP(b_.getInt8Ty(), addr.base, firsts[h]), align,
                                     half(mask, h));
            return;
        }
    }
```

For writes the rule is more restrictive: only the "all chunks contiguous" case is handled statically. A broadcast in a write would mean several lanes writing to the same address, and then the rule "the highest-index lane wins" applies, which the generic scatter path already implements.

### Rematerialization of cheap values

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

A value can be recomputed at the point of use if it is formed only by integer binary operations and conversions over warp-invariant leaves (`threadIdx`, `blockIdx`, parameters, constants; the `is_invariant` criterion) or over uniform values, which already exist as scalars. Divisions and remainders are left out: cloning the computation into a block where the original would not execute could introduce a division exception (`#DE`, delivered as `SIGFPE`) that the original program did not have. Floating-point operations are also left out. The depth is limited to 8 so as not to clone large expressions, whose recomputation would cost more than the load.

## When it applies (and when it does not)

It applies when:

- the kernel was compiled by the JIT with the block shape known, so that `threadIdx` becomes constant per lane (or "uniform + constant", when the block is larger than the variant);
- the index is made of additions, subtractions, `or disjoint`, multiplications by a constant or by a uniform, and shifts by a constant;
- the conversions along the way can be traversed (no overflow, proven or accepted).

It does not apply when:

- the module is the generic one, with no known block shape: `threadIdx` is read from memory and is not constant;
- the index depends on something that varies in an unknown way per lane (a value read from memory, as in `x[idx[i]]` in a sparse matrix, or the product of two varying values);
- the pattern, even if known, is neither contiguous nor broadcast (for example an access with stride `N`): in that case it is proven that there is no shortcut and the path with tests continues to be used.

The same `static_chunks` is also used by the explicit prefetches (`src/compiler/simt.cpp:2343-2346`): if the lanes of a chunk are together, one prefetch per chunk is enough.

## Results

- 256-lane variant of the tiled matmul: from **8108 to 549 instructions**, and from several pattern tests to **0**.
- Tiled matmul: from **22.5 to 14.8 Gcyc** (billions of cycles). The naive one takes 21.4 Gcyc, so the tiled one came to win by a wide margin, as happens on a real GPU. Before, it lost to the naive one.
- 128-lane variant: from **4689 to 2997 instructions**.
- Naive matmul: without the "single row" case, it had regressed to about **24 Gcyc**; with it, it went back to **21.4 Gcyc**.

There was no isolated measurement of the gain from rematerialization (`remat_ok`); it came in together with the static analysis.

## Limitations and trade-offs

- The analysis only proves two patterns (contiguous and broadcast). An access with a constant stride other than 1 element, for example, is known but falls to the path with tests.
- For writes, only the "all contiguous" case is static.
- Traversing `sext` in unflagged sums is an assumption (the same as `split_index`): an `int` index that overflows within a valid allocation would give a result different from CUDA. In practice this would require an allocation of billions of elements.
- Rematerialization trades memory for recomputation; for long expressions the recomputation could cost more than a load, hence the depth limit.

## See also

- [00-basic-concepts.md](../00-basic-concepts.md): lanes, warps, shared memory, cache.
- [01-specialized-jit.md](../iteration-1/01-specialized-jit.md): where the constant `threadIdx` comes from.
- [02-width-variants.md](../iteration-1/02-width-variants.md): the 32, 64, 128 and 256-lane variants.
- [05-address-decomposition.md](../iteration-1/05-address-decomposition.md): `split_address` and `split_index`, which produce the offsets analyzed here.
- [06-dynamic-coalescing.md](../iteration-1/06-dynamic-coalescing.md): the path with run-time tests, used when the proof fails.
- [09-state-at-barriers.md](../iteration-1/09-state-at-barriers.md): the frame where live values cross barriers.
- [13-no-overflow-proof.md](13-no-overflow-proof.md): how to traverse `zext` without the `nuw` flag.
- [22-what-did-not-work.md](22-what-did-not-work.md): the idea of `mmap` with prefetch for the shared memory.
