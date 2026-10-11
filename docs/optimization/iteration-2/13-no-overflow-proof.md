# 13. No-overflow proof

> **In one sentence:** by marking the `blockIdx` registers with the limits that CUDA itself guarantees and asking LLVM "with these value ranges, can this sum wrap around?", the compiler can prove that indices computed in unsigned 32 bits do not overflow, and with this the static access analysis ([12](12-static-access-analysis.md)) also starts working on kernels that use `size_t`.

## The problem

The iso2dfd benchmark (from HeCBench, a collection of reference CUDA programs) computes its indices like this:

```cpp
size_t gidRow = blockDim.y * blockIdx.y + threadIdx.y;
```

The three operands are `unsigned int`, so the computation is done in 32 bits modulo 2^32 and only afterwards is the result converted to `size_t`. In the IR this is an `add`/`mul` in `i32` followed by a `zext` to `i64`.

The static analysis of document [12](12-static-access-analysis.md) needs to distribute the `zext` over the sum, replacing `zext(a + b)` with `zext(a) + zext(b)`. The equality only holds if the 32-bit sum did not wrap around. Counterexample in 8 bits:

```text
a = 200, b = 100 (8 bits unsigned, maximum 255)
a + b in 8 bits = 300 - 256 = 44   ->  zext(44) = 44
zext(a) + zext(b) in 64 bits = 300
44 != 300
```

For this reason the analysis only traversed the `zext` if the sum had the `nuw` flag (*no unsigned wrap*). Semantics in LLVM: if the operation with `nuw` wraps in the unsigned sense, the result is poison; the flag is therefore a license for the optimizer to assume there is no wrap. The frontend can only emit it when the language makes the wrap impossible or undefined. In C and C++, signed overflow is undefined behavior (and Clang emits `nsw` for `int`), but unsigned arithmetic is defined as modulo 2^n. Clang, therefore, never puts `nuw` on `unsigned` sums.

Consequence in iso2dfd: the proof failed, the compiler fell back to the path with run-time tests ([06-dynamic-coalescing.md](../iteration-1/06-dynamic-coalescing.md)), and the 32-lane kernel ended up with **16 test blocks and 32 gathers**, totaling **656 instructions**.

## The idea

The sum can wrap around in type `i32`, but not with the values that the operands can take. CUDA imposes limits on the grid of blocks:

| Register | CUDA limit |
|---|---|
| `blockIdx.x` | less than 2^31 - 1 |
| `blockIdx.y` | less than 65535 |
| `blockIdx.z` | less than 65535 |

And in the JIT the block size is constant (for example `blockDim.y = 16`) and `threadIdx.y` is also known per lane ([01-specialized-jit.md](../iteration-1/01-specialized-jit.md)). Then:

```text
blockDim.y * blockIdx.y + threadIdx.y
<= 16 * 65534 + 15
=  1,048,559
```

well below 2^32 - 1 = 4,294,967,295. The wrap is impossible; the only thing missing is for the compiler to know it.

The fix has two parts:

1. **Declare the ranges to LLVM.** The `blockIdx` loads in the kernel prologue get `!range !{i32 0, i32 limit}` metadata, which asserts that the loaded value is in the half-open interval `[0, limit)`. A value outside the range violates the annotation (the load's result stops being defined), so the optimizer can assume the range as a fact.
2. **Query ValueTracking.** When the `nuw` flag is missing, `linear_form` calls `computeOverflowForUnsignedAdd` and `computeOverflowForUnsignedMul`, from LLVM's ValueTracking analysis. Both derive an interval (`ConstantRange`) for each operand and test whether the operation can leave `[0, 2^32)`:
   - for the sum, each operand's interval comes from `computeConstantRangeIncludingKnownBits`, which combines range analysis (which reads `!range`, constants and operations such as `and`/`lshr`/`udiv` with a constant) with the *known bits* (`computeKnownBits`, which propagates which bits are certainly 0 or 1); the test is `unsignedAddMayOverflow`, which compares the maxima of the two intervals with `2^32 - 1`;
   - for the multiplication, the intervals come only from the known bits (`ConstantRange::fromKnownBits`), and the test is `unsignedMulMayOverflow`. Since known bits only describe intervals with an upper bound of the form `2^k - 1`, the multiplication analysis is coarser: `[0, 65535)` becomes `[0, 65535]`, with no loss in this case, but a limit such as 1000 becomes 1023.

   The answer is an `OverflowResult`: `NeverOverflows`, `MayOverflow`, `AlwaysOverflowsLow` or `AlwaysOverflowsHigh`. Only the first authorizes distributing the `zext`.

## How it works, step by step

The example index, with a 16x16 block in the 32-lane variant:

```text
gidRow = 16 * blockIdx.y + threadIdx.y        (32 bits, unsigned)
idx    = zext(gidRow) ...                       (64 bits)
```

1. `linear_form` finds the `zext` and starts analyzing `gidRow` remembering that it is "under a zext".
2. `gidRow` is a sum without `nuw`. Before, the analysis gave up here.
3. Now it queries `computeOverflowForUnsignedAdd(16 * blockIdx.y, threadIdx.y)`:
   - the range of `blockIdx.y` is `[0, 65535)`, by the annotation;
   - hence `16 * blockIdx.y` is in `[0, 1,048,544]`. The multiplication, when visited recursively by `linear_form`, is also under the `zext` and goes through the same check, with `computeOverflowForUnsignedMul`, which answers `NeverOverflows`;
   - `threadIdx.y` is a vector of constants in `[0, 15]`;
   - the maximum of the sum is 1,048,559 < 2^32.
4. The answer is `NeverOverflows`. The analysis proceeds as if there were `nuw`, distributing the `zext` over the operands.
5. With the complete linear form, `static_chunks` proves that each 16-lane chunk is contiguous and the access becomes a direct vector load.

If the kernel had, for example, `blockIdx.x * 1000000 + threadIdx.x`, the range of `blockIdx.x` (up to 2^31 - 2) multiplied by 10^6 would exceed 2^32, the answer would be `MayOverflow` and the compiler would stay on the path with tests. This is the correct behavior: in this case the wrap is really possible.

## The code

### The ranges in the prologue

`src/compiler/simt.cpp:1956-1966`

```cpp
    // blockIdx within CUDA's grid limits: lets the optimizer (and
    // linear_form) prove that index arithmetic built on it cannot wrap.
    auto ranged = [&](Value* v, uint32_t limit) {
        cast<Instruction>(v)->setMetadata(
            LLVMContext::MD_range,
            MDNode::get(ctx_, {ConstantAsMetadata::get(b_.getInt32(0)), ConstantAsMetadata::get(b_.getInt32(limit))}));
        return v;
    };
    special_[(int)SpecialReg::CtaidX] = ranged(load_u32(CPUSIM_CTX_BLOCK_IDX, "ctaid.x"), 0x7fffffff);
    special_[(int)SpecialReg::CtaidY] = ranged(load_u32(CPUSIM_CTX_BLOCK_IDX + 4, "ctaid.y"), 65535);
    special_[(int)SpecialReg::CtaidZ] = ranged(load_u32(CPUSIM_CTX_BLOCK_IDX + 8, "ctaid.z"), 65535);
```

The prologue is the section at the start of the generated function that reads the special registers, such as `blockIdx`, from the warp context (the structure that the runtime fills in per warp). The `ranged` function attaches to the load a `!range` with the interval `[0, limit)` (exclusive upper bound). The limits are CUDA's: `0x7fffffff` (2^31 - 1) for `x` and 65535 for `y` and `z`. The metadata generates no code; it only constrains what the analyses can assume about the value, and is used both by `linear_form` and by LLVM's normal passes (InstCombine, for example, may start inferring `nuw`/`nsw` on derived computations).

### The question to LLVM inside `linear_form`

`src/compiler/simt.cpp:1026-1042`

```cpp
    bool disjoint = bo->getOpcode() == Instruction::Or && cast<PossiblyDisjointInst>(bo)->isDisjoint();
    if (ext == 1 && !disjoint) {
        auto* ob = dyn_cast<OverflowingBinaryOperator>(bo);
        if (!ob) return false;
        if (!ob->hasNoUnsignedWrap()) {
            // No flag, but the value ranges may still rule out wrapping
            // (blockIdx.y * 16 + threadIdx.y with size_t indexes).
            SimplifyQuery q(m_.getDataLayout());
            OverflowResult r = OverflowResult::MayOverflow;
            if (bo->getOpcode() == Instruction::Add)
                r = computeOverflowForUnsignedAdd(WithCache<const Value*>(bo->getOperand(0)),
                                                  WithCache<const Value*>(bo->getOperand(1)), q);
            else if (bo->getOpcode() == Instruction::Mul)
                r = computeOverflowForUnsignedMul(bo->getOperand(0), bo->getOperand(1), q);
            if (r != OverflowResult::NeverOverflows) return false;
        }
    }
```

`ext == 1` means "we are under a `zext`". In that case:

- an `or disjoint` always passes (the operands have no 1 bits in common, so it is equal to the sum and there is no carry);
- an operation that is not an `OverflowingBinaryOperator` (the class of those that can carry `nuw`/`nsw`: `add`, `sub`, `mul`, `shl`) makes the analysis give up;
- if the `nuw` flag is present, it passes directly;
- if it is not, a `SimplifyQuery` is built and `computeOverflowForUnsignedAdd` or `computeOverflowForUnsignedMul` is queried. Only `NeverOverflows` releases the operation.

The `SimplifyQuery` here is built only with the module's `DataLayout`: no `DominatorTree`, no `AssumptionCache` and no context instruction. Practical consequence: the analysis uses only what is deducible from the value graph itself (constants, `!range`, known bits propagated through the operations) and ignores conditional facts, such as `llvm.assume` or a comparison that dominates the use point (`if (i < n)`).

A `sub` without `nuw` under a `zext` is not queried: it keeps the initial value `MayOverflow` and the analysis gives up. The same holds for `shl`. Only `add` and `mul` are queried.

### Why `threadIdx` did not need to be touched

In the variant specialized by the JIT, when the block fits in the variant, `threadIdx` is a vector of constants (`src/compiler/simt.cpp:1535-1538`), whose known bits are exact. When the block is larger than the variant, the warp index also gets a `!range` annotation in `emit_specialized_tid`:

`src/compiler/simt.cpp:1553-1558`

```cpp
    Value* warp = b_.CreateAlignedLoad(i32, b_.CreateConstInBoundsGEP1_64(b_.getInt8Ty(), ctx_arg_, CPUSIM_CTX_WARP_INDEX),
                                       Align(4), "warp");
    cast<Instruction>(warp)->setMetadata(
        LLVMContext::MD_range, MDNode::get(ctx_, {ConstantAsMetadata::get(b_.getInt32(0)),
                                                  ConstantAsMetadata::get(b_.getInt32((threads + 31) / 32))}));
    Value* base = b_.CreateMul(warp, b_.getInt32(CPUSIM_WARP_SIZE), "base", true, true);
```

The same principle: the warp index within the block is in `[0, ceil(threads / 32))`, and the annotation allows ValueTracking to bound `base = warp * 32` and the sums with the per-lane offset. Note that here the multiplication is already emitted with `nuw` and `nsw` (the two `true` arguments of `CreateMul`), which makes the query unnecessary for it.

## When it applies (and when it does not)

It applies when:

- the index is computed in unsigned 32 bits and then extended with `zext` (the typical pattern of code that uses `size_t` but does the computation with `unsigned`);
- the operands have ranges that LLVM can deduce: constants, specialized `threadIdx`, annotated `blockIdx`, or products and sums of these.

It does not apply when:

- one of the operands is a kernel parameter (for example the matrix width `N`) with no known bound: the known bits of `N` are all unknown, `N * blockIdx.y` may wrap, and the proof fails. The code remains correct, only on the path with tests;
- the operation is `sub`, `shl` or another that is not `add`/`mul`;
- the bound is only guaranteed by a dominating condition (`if (i < n)`), which the context-free `SimplifyQuery` does not see;
- the kernel is the generic module, with no block shape: `threadIdx` is read from memory with no range annotation.

Under `sext` (sign extension), no proof is required; the justification is in document [12](12-static-access-analysis.md) and in [05-address-decomposition.md](../iteration-1/05-address-decomposition.md).

## Results

- 32-lane kernel of iso2dfd: from **656 to 119 instructions**, from 32 gathers to **0**.
- iso2dfd time: from about **1.41 ms to 1.27 ms**.

These numbers are from before the change in the distribution of blocks among the workers ([17-block-distribution.md](17-block-distribution.md)), which later took iso2dfd to 0.38 to 0.42 ms.

## Limitations and trade-offs

- The proof depends on ValueTracking being able to derive the ranges. The recursion of `computeKnownBits` has a fixed maximum depth (6 in LLVM 19), and values loaded from memory without `!range` have no range; in those cases the proof fails (without error, only without gain).
- The limits used are CUDA's. If a program launched a grid larger than CUDA allows, the annotation would be violated and the optimized code could become incorrect; such a program, however, already fails at launch on a real GPU.
- The ValueTracking query has a compile-time cost, small and restricted to the cases with no `nuw` flag under a `zext`.

## See also

- [00-basic-concepts.md](../00-basic-concepts.md): lanes, warps, blocks and grid.
- [01-specialized-jit.md](../iteration-1/01-specialized-jit.md): why `threadIdx` becomes constant.
- [05-address-decomposition.md](../iteration-1/05-address-decomposition.md): the separation between the uniform part and the per-lane part of the address.
- [06-dynamic-coalescing.md](../iteration-1/06-dynamic-coalescing.md): the path with tests, used when the proof fails.
- [12-static-access-analysis.md](12-static-access-analysis.md): the analysis that this proof unlocks.
- [17-block-distribution.md](17-block-distribution.md): the change that took iso2dfd to 0.38 to 0.42 ms.
