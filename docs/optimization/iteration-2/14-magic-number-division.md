# 14. Magic number division

> **In one sentence:** when all lanes divide by the same number (known only when the program runs), the compiler computes a single "magic number" from the divisor once and replaces each division with a multiplication and a few bit shifts, operations that AVX-512 performs on 16 lanes at a time.

## The problem

The heat2d benchmark (from HeCBench, a collection of reference CUDA programs) simulates heat diffusion on a 2D grid. Each thread receives a number `i` and finds its position in the grid as follows:

```cpp
x = i % Lx;     // column (remainder of the division)
y = i / Lx;     // row (quotient)
```

and then computes the neighbors with expressions such as `(x + 1) % Lx` and `(y + 1) % Ly` (periodic boundary condition). `Lx` and `Ly` are kernel parameters: they are only known when the program runs.

AVX-512 **has no vector integer division** (there are `vdivps`/`vdivpd` for floating point, but nothing equivalent for integers). When legalizing a `udiv <16 x i32>`, the x86 backend scalarizes: for each lane, it extracts the element, executes a scalar `div` and reinserts the result into the vector. Each 32-bit `div` costs something like 20 to 40 cycles, and the divisions of the lanes are serialized on the same divider unit.

In heat2d, there were 5 division or remainder operations per 32-lane call. Result: **5 x 32 = 160 scalar divisions per call**. This is what dominated the kernel time, which took about 9.8 ms per iteration, slower than the OpenMP CPU version of the same benchmark (9.23 ms).

## The idea

Division by a fixed `d` can be replaced by a multiplication by the reciprocal in fixed point: an integer `m` close to `2^32 / d`, and then

```text
n / d  ~  (n * m) / 2^32
```

The division by `2^32` comes for free: it is the high 32 bits of the 64-bit product (*mulhi*). With only 32 bits of precision in the reciprocal, however, the rounding error makes the result incorrect for some `n`. The exact form, described below, uses a 33-bit multiplier `M ~ 2^(32+l) / d` and a final shift by `l`. The non-trivial point is choosing `M` and the sequence of operations so that the result is **exact** (equal to `floor(n / d)`) for every 32-bit `n`, and not merely approximate. Multiplication, unlike division, exists in vector form: AVX-512 has no unsigned 32-bit *mulhi*, but `vpmuludq` (32 x 32 -> 64-bit product of the low 32 bits of each 64-bit element) solves it: since both factors are 32-bit `zext`s, the backend implements `trunc(lshr(mul(zext a, zext b), 32))` with `vpmuludq` and shuffles, without needing a full 64 x 64 multiplication.

Compilers already do this when the divisor is a known constant (`x / 7` in the source code). What is new here is doing the same when the divisor is only known at run time, but is **uniform**: equal for all lanes (see [00-basic-concepts.md](../00-basic-concepts.md)). The magic number is computed once, with one 64-bit scalar division, and reused by all lanes and, when the division is inside a loop, by all iterations.

The technique is that of Granlund and Montgomery (1994), in the form called "round-up", which is exact for every unsigned 32-bit divisor.

## How it works, step by step

### The formulas

For a divisor `d` (between 1 and 2^32 - 1), compute once:

```text
l = ceil(log2 d)                         (how many bits are needed for d - 1)
m = floor(2^32 * (2^l - d) / d) + 1      (the magic number, fits in 32 bits)
sh1 = min(l, 1)
sh2 = max(l - 1, 0)
```

And for each dividend `n`, in each lane:

```text
t = mulhi(m, n)                          (the upper 32 bits of m * n)
q = (t + ((n - t) >> sh1)) >> sh2        (the exact quotient)
remainder = n - q * d
```

The term `(n - t) >> 1` exists because the effective multiplier is `M = 2^32 + m`, which needs 33 bits. Since `M * n / 2^32 = n + m * n / 2^32`, the quotient is `floor((n + t) / 2^l)`; but `n + t` can exceed 2^32. The expression `t + ((n - t) >> 1)` computes `floor((n + t) / 2)` without ever forming `n + t`, and the remaining shift is `l - 1`.

### Proof of exactness

Let `N = 32`, `1 <= d < 2^N`, `l = ceil(log2 d)` (so `2^(l-1) < d <= 2^l` for `d >= 2`) and `M = 2^N + m`. By the definition of `m`:

```text
M = 2^N + floor(2^N * (2^l - d) / d) + 1 = floor(2^(N+l) / d) + 1
```

since `2^N * (2^l - d) / d = 2^(N+l) / d - 2^N` and `2^N` is an integer. Hence `M = (2^(N+l) + e) / d` for some `0 < e <= d`.

**Step 1: `floor(M * n / 2^(N+l)) = floor(n / d)` for every `0 <= n < 2^N`.** Write `n = q*d + r` with `0 <= r < d`. Then

```text
M * n / 2^(N+l) = n/d + e*n / (d * 2^(N+l)) = q + (r + e*n / 2^(N+l)) / d
```

The error term `e*n / 2^(N+l)` is non-negative and smaller than 1, because `e <= d <= 2^l` and `n < 2^N`. Then `0 <= r + e*n/2^(N+l) < r + 1 <= d`, the fraction in parentheses divided by `d` lies in `[0, 1)`, and the floor is exactly `q`.

**Step 2: the instruction sequence computes this floor.** With `t = floor(m * n / 2^N)`, and using `floor((a + x) / k) = floor((a + floor(x)) / k)` for integers `a`, `k`:

```text
floor(M * n / 2^(N+l)) = floor((n + m*n/2^N) / 2^l) = floor((n + t) / 2^l)
```

For `l >= 1`, `floor((n + t) / 2^l) = floor(floor((n + t) / 2) / 2^(l-1))`, and `floor((n + t) / 2) = t + floor((n - t) / 2)`. Since `m < 2^N` (see below), `t <= n`, so `n - t` does not wrap around, and `t + ((n - t) >> 1) <= n < 2^N` does not either. Hence `q = (t + ((n - t) >> 1)) >> (l - 1)`, which is the formula with `sh1 = 1` and `sh2 = l - 1`. For `d = 1` (`l = 0`), `m = 1`, `t = 0`, `sh1 = sh2 = 0` and the formula returns `t + (n - t) = n`.

**`m` fits in 32 bits.** Since `d > 2^(l-1)`, `(2^l - d) / d < 1`; the maximum of this ratio, at `d = 2^(l-1) + 1`, is `1 - 2 / (2^(l-1) + 1)`, and for `l <= 32` this is smaller than `1 - 2^-N`. Then `floor(2^N * (2^l - d) / d) <= 2^N - 2` and `m <= 2^N - 1`. In the code, the numerator `(2^l - d) << 32` also fits in 64 bits: for `l = 32`, `2^l - d < 2^31`.

### Worked example: n = 100, d = 7

Computing the magic number (once):

```text
l   = ceil(log2 7) = 3                         (2^2 = 4 < 7 <= 8 = 2^3)
m   = floor(2^32 * (2^3 - 7) / 7) + 1
    = floor(4294967296 * 1 / 7) + 1
    = floor(613566756.57...) + 1
    = 613566757
sh1 = min(3, 1) = 1
sh2 = max(3 - 1, 0) = 2
```

Per-lane computation, with `n = 100`:

```text
m * n = 613566757 * 100 = 61356675700
t     = floor(61356675700 / 2^32)
      = floor(61356675700 / 4294967296)
      = 14                                     (14 * 4294967296 = 60129542144 <= 61356675700)
n - t = 100 - 14 = 86
86 >> 1 = 43
t + 43  = 57
q = 57 >> 2 = 14                               (57 / 4 = 14.25)
remainder = 100 - 14 * 7 = 2
```

Checking: 100 / 7 = 14 with remainder 2. Correct.

### Boundary example: n = 4294967295 (the largest 32-bit value), d = 7

```text
t     = mulhi(613566757, 4294967295) = 613566756
n - t = 4294967295 - 613566756 = 3681400539
>> 1  = 1840700269
t + 1840700269 = 2454267025
q     = 2454267025 >> 2 = 613566756
```

And indeed 4294967295 / 7 = 613566756 (with remainder 3). `t + ((n - t) >> 1)` stayed below 2^32, whereas `t + n = 4908534051` would have wrapped around in 32 bits. The computations above were checked with a script, together with 200 thousand random pairs of `n` and `d`.

### Signed division

For `int` (signed), the compiler:

1. saves the sign of the divisor and of each dividend;
2. divides the absolute values with the unsigned formula (`|INT_MIN| = 2^31` is representable as `u32`, so the extreme case needs no separate handling);
3. negates the quotient if the signs of `n` and `d` were different.

This truncates toward zero, as C and CUDA specify. Example: `-100 / 7`: absolute values 100 and 7, quotient 14, different signs, result `-14`; remainder `-100 - (-14 * 7) = -2`. Same as what C gives.

### Zero divisor

Division by zero is undefined behavior in C and CUDA. The problem is a different one: in the SIMT model, the code of an `if` is executed under a mask even when no active lane enters it, and the computation of the magic number is scalar, outside the mask (and may even be hoisted out of loops by LICM). If the divisor were zero on a path that no lane actually executes, the scalar `div` in the computation of `m` would raise `#DE`, delivered by the kernel as `SIGFPE`, in a program that would be correct in CUDA. For this reason a zero divisor is replaced by 1. If some lane did divide by zero, the result was already undefined; if none did, nothing changes.

## The code

### The entry point

`src/compiler/simt.cpp:2554-2555`

```cpp
    if (auto* bin = dyn_cast<BinaryOperator>(&inst)) {
        if (Value* v = emit_uniform_division(bin)) return set(v);
```

Every binary operation goes through `emit_uniform_division` first. If it returns a value, the division was handled; if it returns `nullptr`, the normal path continues (which, for divisions, puts 1 as the divisor in the inactive lanes to avoid the same `SIGFPE`).

### When it applies

`src/compiler/simt.cpp:1623-1629`

```cpp
Value* WarpBuilder::emit_uniform_division(BinaryOperator* bin) {
    const auto op = bin->getOpcode();
    if (op != Instruction::UDiv && op != Instruction::URem && op != Instruction::SDiv && op != Instruction::SRem)
        return nullptr;
    Value* divisor = bin->getOperand(1);
    if (!bin->getType()->isIntegerTy(32) || uni_->is_varying(divisor) || isa<Constant>(divisor)) return nullptr;
    const bool is_signed = op == Instruction::SDiv || op == Instruction::SRem;
```

Only 32-bit divisions and remainders (`UDiv`, `URem`, `SDiv`, `SRem`) whose divisor is uniform and **not** constant are handled. A constant divisor is left out because LLVM itself already makes this replacement, with even better formulas chosen for that specific number.

### Computing the magic number (once)

`src/compiler/simt.cpp:1635-1649`

```cpp
    // Scalar setup (uniform; leaves loops through LICM).
    Value* d = scalar(divisor);
    Value* neg_d = is_signed ? b_.CreateICmpSLT(d, b_.getInt32(0)) : nullptr;
    Value* ad = is_signed ? b_.CreateSelect(neg_d, b_.CreateNeg(d), d) : d;  // |d| as unsigned
    ad = b_.CreateSelect(b_.CreateICmpEQ(ad, b_.getInt32(0)), b_.getInt32(1), ad);
    Value* ctlz = b_.CreateBinaryIntrinsic(Intrinsic::ctlz, b_.CreateSub(ad, b_.getInt32(1)), b_.getTrue());
    // ctlz(0) is 32 (poison flag off below): d = 1 gives l = 0.
    ctlz = b_.CreateSelect(b_.CreateICmpEQ(ad, b_.getInt32(1)), b_.getInt32(32), ctlz);
    Value* l = b_.CreateSub(b_.getInt32(32), ctlz);
    Value* l64 = b_.CreateZExt(l, i64);
    Value* ad64 = b_.CreateZExt(ad, i64);
    Value* top = b_.CreateShl(b_.CreateSub(b_.CreateShl(b_.getInt64(1), l64), ad64), b_.getInt64(32));
    Value* m = b_.CreateTrunc(b_.CreateAdd(b_.CreateUDiv(top, ad64), b_.getInt64(1)), i32);
    Value* sh1 = b_.CreateSelect(b_.CreateICmpUGT(l, b_.getInt32(0)), b_.getInt32(1), b_.getInt32(0));
    Value* sh2 = b_.CreateSub(l, sh1);
```

Everything here is scalar, done once:

- `ad` is the absolute value of the divisor, and becomes 1 if it is zero (the protection against `SIGFPE`).
- `l = 32 - ctlz(ad - 1)`: `ctlz` (count leading zeros; `lzcnt` on x86) gives `ceil(log2 ad)`. For `d = 7`: `ad - 1 = 6 = 110` in binary, 29 leading zeros, `l = 3`. The case `d = 1` is handled separately (`ad - 1 = 0`, so `l = 0`).
- `top = (2^l - ad) << 32` and `m = top / ad + 1`: this is the magic number formula, with **the only scalar division** of the whole operation.
- `sh1` and `sh2` are the two shifts (`min(l,1)` and `l - sh1 = max(l-1, 0)`).

The comment "leaves loops through LICM" means: since everything depends only on the divisor, which is loop-invariant, LLVM's LICM pass moves this block to the loop preheader.

### The per-lane computation

`src/compiler/simt.cpp:1651-1670`

```cpp
    // Per lane.
    Value* n = vector(bin->getOperand(0));
    Value* neg_n = nullptr;
    Value* un = n;
    if (is_signed) {
        neg_n = b_.CreateICmpSLT(n, Constant::getNullValue(v32));
        un = b_.CreateSelect(neg_n, b_.CreateNeg(n), n);
    }
    Value* t = b_.CreateTrunc(
        b_.CreateLShr(b_.CreateMul(b_.CreateZExt(un, v64), b_.CreateVectorSplat(W, b_.CreateZExt(m, i64))),
                      ConstantVector::getSplat(ElementCount::getFixed(W), b_.getInt64(32))),
        v32);
    Value* q = b_.CreateLShr(b_.CreateAdd(t, b_.CreateLShr(b_.CreateSub(un, t), b_.CreateVectorSplat(W, sh1))),
                             b_.CreateVectorSplat(W, sh2));
    if (is_signed) {
        Value* flip = b_.CreateXor(neg_n, b_.CreateVectorSplat(W, neg_d));  // signs differ
        q = b_.CreateSelect(flip, b_.CreateNeg(q), q);
    }
    if (op == Instruction::UDiv || op == Instruction::SDiv) return q;
    return b_.CreateSub(n, b_.CreateMul(q, b_.CreateVectorSplat(W, d)), bin->getName());
```

Now everything is vector (`W` lanes at a time):

- `un` is the absolute value of each dividend (only in the signed case).
- `t`: extends to 64 bits, multiplies by the splat of `m` and takes the high 32 bits (`>> 32`). This is the *mulhi*, implemented with `vpmuludq` as explained in "The idea".
- `q`: exactly the formula `(t + ((n - t) >> sh1)) >> sh2`.
- In the signed case, `flip = neg_n xor neg_d` marks the lanes where the signs differ, and in them the quotient is negated.
- For the remainder: `n - q * d`. Note that it uses the original `d` (signed) and the original `n`, which gives the remainder with the correct C sign.

## When it applies (and when it does not)

It applies when:

- the operation is `/` or `%` on 32-bit integers, signed or unsigned;
- the divisor is uniform (the same in all lanes), for example a kernel parameter such as `Lx`, or `blockDim.x` in the generic module;
- the divisor is not a constant (for constants, LLVM already does something equivalent).

It does not apply when:

- the divisor varies per lane (each thread divides by a different number): then there is no single magic number, and the code continues with one scalar division per lane;
- the integers are 64-bit: the formula would need the high half of a 64 x 64 -> 128-bit product, which AVX-512 does not offer in vector form (the IFMA extension, present on Zen 4, only covers 52-bit products);
- the division is floating point (AVX-512 does that natively).

## Results

- Scalar divisions per 32-lane call in heat2d: from **160 to 4**. The 4 that remain are the magic number computations, done once per call instead of once per lane.
- heat2d: from about **9.8 ms to 4.0 to 4.5 ms per iteration**, that is, **2.2 times faster than the OpenMP version** (9.23 ms).
- New test: `tests/cuda/division.cu` divides extreme dividends (0, 1, 2^31 - 1, 2^31, 2^32 - 1, `INT_MIN`, `INT_MAX` and thousands of pseudorandom values) by extreme divisors (1, 2, 7, 641, 65535, 2^31, 2^32 - 1, negatives, `INT_MIN`), signed and unsigned, and compares with the result computed on the host. It passes in all 12 modes of `tests/cuda_feature_test.sh`: ten width configurations (`auto`, `32`, `64`, `64s`, `128`, `128s`, `128c`, `256`, `256c`, `512`), the generic module without JIT and the daemon backend.

## Limitations and trade-offs

- The magic number computation costs one scalar 64-bit division. If the divisor changed on each iteration of a loop (for example, if it depended on the counter), this division would stay inside the loop; even so, it would be one division instead of 32.
- The case `INT_MIN / -1` (which overflows in C and is undefined) is not handled specially; the test skips this case on purpose.
- The "round-up" form always uses the longest sequence (with the `(n - t) >> 1`). For some divisors a shorter sequence would exist, but choosing between them would require a branch or two versions of the code; since the divisor is only known at run time, the single, always-exact form was preferred.

## See also

- [00-basic-concepts.md](../00-basic-concepts.md): lanes, uniform values and AVX-512.
- [01-specialized-jit.md](../iteration-1/01-specialized-jit.md): the JIT, which turns `blockDim` into a constant (and then it is LLVM that optimizes the division).
- [07-uniform-loops.md](../iteration-1/07-uniform-loops.md): loops in which the magic number computation stays outside.
- [11-llvm-tuning.md](../iteration-1/11-llvm-tuning.md): other points where the generated code was adapted to what LLVM and x86 do well.
- [17-block-distribution.md](17-block-distribution.md): another change that affected the performance of the HeCBench benchmarks.
