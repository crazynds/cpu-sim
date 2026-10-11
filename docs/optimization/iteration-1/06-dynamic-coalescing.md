# 06. Dynamic coalescing of gathers and scatters

> **In one sentence:** when the 32 threads of a warp read or write memory through addresses that can be anything, the generated code checks at run time whether the addresses are consecutive (a single contiguous vector load is enough) or all equal (a scalar load with broadcast is enough), and only uses gather and scatter, which are expensive on Zen 4, when neither pattern holds.

> **Iteration:** created in iteration 1. In [iteration 2](../iteration-2/README.md) an adjustment entered the rare path: in variants with more than 32 lanes, a direct gather per chunk instead of testing three paths (for compile time, see [21](../iteration-2/21-compile-time.md)); and a good share of the accesses came to be resolved at compile time, without going through these tests (see [12](../iteration-2/12-static-access-analysis.md)).

## The problem

With AVX-512 there are three main ways to load 16 `float` values into a zmm (the project's general concepts are in [00-basic-concepts.md](../00-basic-concepts.md)):

| Method | When it works | Approximate cost |
|---|---|---|
| **contiguous load** (`vmovups`) | the 16 values occupy 64 consecutive bytes | 1 instruction, one L1 access (two if it crosses a cache line) |
| **scalar load + broadcast** (`vbroadcastss`) | the 16 lanes use the same address | 1 access of 4 bytes replicated across the 16 lanes |
| **gather** (`vgatherdps`) | each lane has an arbitrary address | decomposed into one load uop per element: 16 L1 accesses competing for the load ports |

For writes there is the equivalent: **contiguous store** and **scatter** (`vscatterdps`). On Zen 4 the scatter is **microcoded**: instead of being decoded into a few uops, it is issued by the microcode sequencer (MSROM) as a long sequence of uops, with one store per element and a mask update at each element, which makes it dozens of times slower than a contiguous store and blocks the decoder while it is being issued. The gather is not microcoded on Zen 4, but its cost grows with the number of elements, since the hardware does not group elements of the same cache line the way a contiguous load does.

For `B[k * N + col]`, the uniformity analysis only says that the address is varying; most of the time the pattern across lanes **is not known at compile time**. The conservative translation would always be gather and scatter. But in CUDA programs the great majority of accesses are:

- **contiguous**: thread `i` reads element `base + i` (what on a GPU is called a **coalesced** access);
- or **same address**: all threads read the same value (for example, `A[row * K + k]`, when the threads of the group are in the same row).

In those cases a gather costs 16 accesses where 1 would suffice.

## The idea

The generated code classifies, at run time, the vector of addresses of the active lanes into one of three patterns (consecutive with stride equal to the element size, all equal, or arbitrary) and dispatches by branch to the cheapest instruction that produces the same result as the gather/scatter. This is called **dynamic coalescing** ("dynamic" because the decision happens during execution, not at compile time).

What makes this cheap is the [address decomposition](05-address-decomposition.md): the tests look only at the lanes' **offsets**, which almost always depend only on `threadIdx` and are therefore loop-invariant. LICM hoists the tests to the preheader; inside the loop what remains is a branch on an invariant condition, which the branch predictor gets right on every iteration (or which loop unswitching turns into separate versions of the loop).

## How it works, step by step

The 32-lane vector is divided into **chunks of 16 lanes** (one `float` zmm register each). For each chunk, the code computes two bit masks from the offsets in bytes. Example with a chunk of 4 lanes and 4-byte elements:

```text
                         lane:   0    1    2    3
offsets (bytes):                 40   44   48   52
delta = offset - offset[0]:      0    4    8   12
expected step (lane * 4):        0    4    8   12
"contiguous" bits (delta == step):  1    1    1    1   -> contiguous!
"same" bits       (delta == 0):     1    0    0    0
```

```text
offsets (bytes):                 40   40   40   40
delta:                           0    0    0    0
"contiguous" bits:               1    0    0    0
"same" bits:                     1    1    1    1   -> same address!
```

```text
offsets (bytes):                 40  400   12   88
"contiguous" bits:               1    0    0    0
"same" bits:                     1    0    0    0   -> neither: gather
```

Then, for each chunk:

- **all lanes contiguous:** a **masked load** (`vmovups zmm{k}{z}, [base]`) from the first address, with the mask of active lanes in a `k` register (on active lanes in the SIMT linearization, see [07-uniform-loops.md](07-uniform-loops.md));
- **all at the same address:** a scalar load and a **broadcast** to the 16 lanes;
- **otherwise:** a real gather.

For writes, the logic is similar, but with only two cases: contiguous becomes a **masked store**, and anything else becomes a **scatter**. There is no "same address" case for writes, because when several lanes write to the same place the rule is **"the last lane wins"** (the value that remains is that of the highest-numbered lane), and the AVX-512 scatter has that semantics by specification: the elements are written in order from lane 0 to 15, and when two indices coincide the write of the highest lane is the one that remains.

### Inactive lanes are never read

An inactive lane can have an invalid address (for example, a thread outside an `if (i < n)` because it is past the end of the array). Reading that address could cause a page fault (SIGSEGV) that would not happen on the GPU. Therefore:

- contiguous loads and stores are **masked**: in AVX-512, elements with bit 0 in the mask have *fault suppression*, that is, they raise no exception and do not access memory, even if the corresponding address is in an unmapped page;
- the pattern tests only require the property **on the active lanes**. The "all contiguous" test is, in fact, "no active lane fails to be contiguous", that is, `(mask AND NOT bits) == 0`.

### Refinements

The basic mechanism was improved in several stages.

**1. Test bits in integer registers.** AVX-512 has **8 mask registers** (`k0` to `k7`, 64 bits each), and `k0` cannot be used as a predicate (encoded as a predicate it means "no mask"), leaving 7. A vector comparison (`vpcmpq`) writes its result into a `k`. In a loop with several accesses, each with its "contiguous" and "same" bits live throughout the loop, the `k` registers ran out. Since the result of a comparison is cheap to recompute, LLVM's register allocator preferred to **rematerialize** the comparisons (`vpcmpneqq`) at every iteration instead of spilling, nullifying the gain of hoisting the tests. The fix was to convert the bits to an `i16` (the `bitcast <16 x i1> to i16` becomes a `kmovw` to a GPR), keep it in one of the 16 GPRs and do the combination with the active-lane mask using scalar `and`/`andn`/`test`. In the naive matmul this took the time from **2340 to 1265 ms**.

**2. Test the whole access first.** Before looking chunk by chunk, the code tests patterns that hold for the whole access:

```text
Test                                          What it generates
--------------------------------------------  -------------------------------------------
all chunks contiguous and with the same        1 load of 16 values, replicated to
first address                                  all chunks
all chunks contiguous                          1 masked load per chunk, no tests
all chunks of single address                   1 broadcast per chunk
none of the above                              the chunk-by-chunk path
```

The first case looks strange, but it is very common. In a block of 16 x 16 threads, a warp of 32 threads covers two rows of the block (`threadIdx.y` = 0 and 1). In the read `Bs[k][threadIdx.x]` of the shared-memory multiplication, the two rows read exactly the same 16 values. A single load serves the 32 lanes.

**3. Broadcast without a branch.** In the "all of single address" case, a chunk with no active lane can have an invalid address, and a scalar load has no mask. Instead of a branch (which, depending on divergence, would be mispredicted), the address becomes `select(any_active, address, context)`: if there is no active lane, a value from the **warp context** (`cpusim_warp_ctx`, always mapped) is read, and the value read is discarded. The `select` becomes a `cmov`, with no control dependence.

**4. Offsets only on the rare path.** The 64-bit offsets of 32 lanes occupy four zmm. The common paths only need the first offset of each chunk and the test bits. Therefore the full vector of offsets is **recomputed inside** the gather path, instead of being kept, freeing registers in the loop.

**5. Too many chunks: direct gather.** Variants wider than a warp (64 or 128 lanes per call, see [02-width-variants.md](02-width-variants.md)) have 4 or 8 chunks. On the rare path, testing each chunk with its 3 paths (contiguous, same address, gather) multiplies the code size by the number of chunks, for each access. In the generic `fdtd3d` kernel (compiled without knowing the block shape), this made compilation take 24 s. Today, with more than 2 chunks, the rare path does **one direct gather per chunk**, and compilation dropped to 7 s. The common cases (all contiguous, all broadcast) are still tested for the whole access before getting there.

## The code

The pattern tests, computed once for the access:

`src/compiler/simt.cpp:1304-1327`

```cpp
    std::vector<Value*> offs(chunks), firsts(chunks), bases(chunks), masks(chunks), cbits(chunks), sbits(chunks);
    Value* all_contig = b_.getTrue();
    Value* all_bcast = b_.getTrue();
    Value* same_first = b_.getTrue();
    for (unsigned h = 0; h < chunks; ++h) {
        offs[h] = half(addr.offsets, h);
        masks[h] = half(mask, h);
        Value* first = b_.CreateExtractElement(offs[h], uint64_t(0));
        Value* delta = b_.CreateSub(offs[h], b_.CreateVectorSplat(16, first));
        cbits[h] = b_.CreateFreeze(b_.CreateBitCast(b_.CreateICmpEQ(delta, lane_steps(b_, size)), i16));
        sbits[h] = b_.CreateFreeze(b_.CreateBitCast(b_.CreateICmpEQ(delta, Constant::getNullValue(delta->getType())), i16));
        all_contig = b_.CreateAnd(all_contig, b_.CreateICmpEQ(cbits[h], b_.getInt16(-1)));
        all_bcast = b_.CreateAnd(all_bcast, b_.CreateICmpEQ(sbits[h], b_.getInt16(-1)));
        if (h > 0) same_first = b_.CreateAnd(same_first, b_.CreateICmpEQ(first, firsts[0]));
        firsts[h] = first;
    }
    all_contig = cached(all_contig);
    all_bcast = cached(all_bcast);
    same_first = cached(same_first);
    for (unsigned h = 0; h < chunks; ++h) {
        bases[h] = b_.CreateGEP(b_.getInt8Ty(), addr.base, cached(firsts[h]));
        cbits[h] = cached(cbits[h]);
        sbits[h] = cached(sbits[h]);
    }
```

For each chunk `h` of 16 lanes:

- `first` is the offset of the first lane and `delta` is the distance from each lane to it;
- `cbits` are the "contiguous" bits: `delta` equal to `lane * size` (`lane_steps` builds the vector `0, size, 2*size, ...`). The `BitCast` to `i16` is refinement 1: it turns the 16 comparisons into a 16-bit integer;
- `sbits` are the "same address" bits: `delta` equal to zero;
- `all_contig`, `all_bcast` and `same_first` accumulate the whole-access tests (refinement 2). Note that they require **all** bits set (`-1` in `i16` is `0xFFFF`), without looking at the mask: thus they do not depend on which lanes are active and can leave the loop;
- `cached` is used in kernels with barriers (`__syncthreads`): the value, which depends only on the warp context, is computed once when the warp starts and stored in its frame, instead of being recomputed at each resume (see [09-state-at-barriers.md](09-state-at-barriers.md)). In kernels without a barrier it does nothing.

The broadcast path without a branch (refinement 3):

`src/compiler/simt.cpp:1365-1379`

```cpp
    b_.SetInsertPoint(bcast);
    {
        auto* v1 = FixedVectorType::get(elem, 1);
        std::vector<Value*> parts(chunks);
        (void)v1;
        for (unsigned h = 0; h < chunks; ++h) {
            // A chunk without active lanes may hold any address: read from the
            // (always readable) context instead and ignore the value. No branch.
            Value* any = b_.CreateICmpNE(b_.CreateBitCast(masks[h], i16), b_.getInt16(0));
            Value* p = b_.CreateSelect(any, bases[h], b_.CreatePointerCast(ctx_arg_, bases[h]->getType()));
            parts[h] = b_.CreateVectorSplat(16, b_.CreateAlignedLoad(elem, p, Align(1)));
        }
        results.push_back({concat(parts), b_.GetInsertBlock()});
        b_.CreateBr(done);
    }
```

`any` says whether the chunk has any active lane. `p` is the real address or, if there is no active lane, the context pointer (`ctx_arg_`). The load always happens, with no `if`, and `CreateVectorSplat` copies the value to the 16 lanes.

The rare path, with the direct gather for many chunks (refinements 4 and 5) and the per-chunk test for 1 or 2 chunks:

`src/compiler/simt.cpp:1381-1407`

```cpp
    b_.SetInsertPoint(mixed);
    {
        // Rare path: rebuild the offsets here rather than keep them alive.
        {
            SplitAddress again = split_address(ptr);
            for (unsigned h = 0; h < chunks; ++h) offs[h] = half(again.offsets, h);
            addr.base = again.base;
        }
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
            Value* m_bits = b_.CreateBitCast(masks[h], i16);
            auto all = [&](Value* bits) { return b_.CreateICmpEQ(b_.CreateAnd(m_bits, b_.CreateNot(bits)), b_.getInt16(0)); };
            BasicBlock* bb_cont = BasicBlock::Create(ctx_, "ld.c.contig", w_);
            BasicBlock* bb_test = BasicBlock::Create(ctx_, "ld.c.test", w_);
            BasicBlock* bb_same = BasicBlock::Create(ctx_, "ld.c.bcast", w_);
            BasicBlock* bb_gather = BasicBlock::Create(ctx_, "ld.c.gather", w_);
            BasicBlock* bb_done = BasicBlock::Create(ctx_, "ld.c.done", w_);
            b_.CreateCondBr(all(cbits[h]), bb_cont, bb_test);
```

- The block between braces at the start calls `split_address` again, only in here: the full 64-bit offsets only exist on this rare path.
- With `chunks > 2` (64- or 128-lane variants), it goes straight to `CreateMaskedGather`, a masked gather per chunk.
- With 1 or 2 chunks, the function `all` does the "active lanes only" test: `m_bits AND NOT bits` is zero when no active lane violates the pattern. The code proceeds to three blocks: contiguous (`ld.c.contig`), broadcast (`ld.c.bcast`) and gather (`ld.c.gather`).

In the scatter, the rare path has the same rule for many chunks:

`src/compiler/simt.cpp:1486-1490`

```cpp
        Value* v = half(vals, h);
        if (chunks > 2) {  // as for loads: one scatter per chunk (lanes in order, the last one wins)
            b_.CreateMaskedScatter(v, b_.CreateGEP(b_.getInt8Ty(), addr.base, offs[h]), align, m);
            continue;
        }
```

The comment recalls the guarantee: the scatter writes the lanes in increasing order, so if two lanes write to the same address, the value of the one with the highest index remains ("the last lane wins").

## When it applies (and when it does not)

- **It applies** to every load and store through a varying address whose pattern was **not** proven at compile time.
- **It does not run when the pattern is known at compile time.** Today the kernel is recompiled at launch knowing the block shape ([01-specialized-jit.md](01-specialized-jit.md)), and then the `threadIdx` of each lane is a constant. In most accesses, the compiler can prove that a chunk is contiguous or of single address and directly generates the right load, with no test at all. This is described in [12-static-access-analysis.md](../iteration-2/12-static-access-analysis.md). Dynamic coalescing remains as a safety net for what the static analysis cannot prove (indices read from memory, the generic module without a block shape, etc.).
- **Uniform accesses** (the same address in all lanes, known at compile time) do not go through here: they become a plain scalar load.
- **For writes** there is no "same address" case: it falls to the scatter, to preserve "the last lane wins".

## Results

- Test bits in integer registers (refinement 1): naive matmul from **2340 ms to 1265 ms**.
- Direct gather per chunk with more than 2 chunks (refinement 5): compilation of the generic `fdtd3d` from **24 s to 7 s**.
- There is no isolated measurement of the basic mechanism (contiguous / broadcast / gather) against "always gather", nor of refinements 2, 3 and 4.

## Limitations and trade-offs

- The tests cost a few instructions. When the offsets change at every iteration (for example, `x[col_idx[j]]`), they cannot leave the loop and are redone every time. In a truly scattered access, one pays for the test and the gather.
- Each access generates several code paths. This increases code size and compile time, and was the reason for refinement 5.
- The broadcast without a branch does a useless read of the context when the chunk is entirely inactive. It is cheap, but not zero.
- The classification is per chunk of 16 lanes: a contiguous access that starts in the middle of a chunk and continues in the other is still detected (each chunk has its own start), but patterns such as "stride 2" (`a[2*i]`) fall to the gather.

## See also

- [00-basic-concepts.md](../00-basic-concepts.md): gather, scatter, mask, active lane.
- [05-address-decomposition.md](05-address-decomposition.md): where the base and the offsets come from.
- [12-static-access-analysis.md](../iteration-2/12-static-access-analysis.md): when the pattern is proven at compile time and none of this runs.
- [02-width-variants.md](02-width-variants.md): the variants with more than 2 chunks.
- [21-compile-time.md](../iteration-2/21-compile-time.md): other measures to compile faster.
- [09-state-at-barriers.md](09-state-at-barriers.md): the per-warp cache used by `cached`.
