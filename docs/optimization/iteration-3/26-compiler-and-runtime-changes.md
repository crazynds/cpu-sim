# 26. Compiler and runtime adjustments

> **In one sentence:** three independent fixes, found during the tensor core work ([25](25-tensor-cores.md)): (a) division by a constant in divergent code is again done with a multiplication and shifts, instead of a scalar `idiv` per lane; (b) in the module compiled ahead of time, wide variants that are too large are dropped and left to the JIT, which cut the compile time of the WMMA test in half; (c) the JIT cache key now includes the identity of the `cpusim-devc` binary, so that a rebuilt compiler does not reuse modules from the previous one.

The general concepts (masks, divergence, generic module, JIT) are in [00-basic-concepts.md](../00-basic-concepts.md).

## (a) Division by a constant in divergent code

### The problem

In divergent code, the SIMT compiler executes both sides of a branch with a mask ([00](../00-basic-concepts.md), [08](../iteration-1/08-if-conversion.md)): the inactive lanes also go through the instructions, with values that do not matter and are discarded afterwards. For an integer division this is dangerous. AVX-512 has no vector integer division, and the backend scalarizes an `sdiv <W x i32>` into one `idiv` per lane ([14](../iteration-2/14-magic-number-division.md)); if the divisor of an inactive lane is 0, the `idiv` raises the #DE exception and the process dies with `SIGFPE`. For this reason the SIMT pass replaced the divisor with `select(mask, d, 1)`: in the inactive lanes, it divides by 1.

The side effect appeared when the divisor was a **constant**. `e / 16` in the per-thread code is an `sdiv` by `i32 16`, which LLVM turns into shifts (with the sign adjustment to round toward zero); a division by any other constant becomes a multiplication by the high half and shifts. After the `select`, the divisor was no longer constant: it was a vector that depended on the mask. LLVM could no longer reduce the division, and each lane did a scalar `idiv`.

In `samples/tensor_gemm.cu`, the copy of the tiles to shared memory (`m = e / 16`, `k = e % 16`, inside a divergent loop) had 640 `idiv` instructions in the hot function, and it was the top of `perf`.

### The idea

A nonzero constant is never zero in any lane, active or inactive. The `select` protects nothing in that case, and can be omitted.

### The code

`src/compiler/simt.cpp:2554-2576`

```cpp
    if (auto* bin = dyn_cast<BinaryOperator>(&inst)) {
        if (Value* v = emit_uniform_division(bin)) return set(v);
        Value* lhs = vector(bin->getOperand(0));
        Value* rhs = vector(bin->getOperand(1));
        switch (bin->getOpcode()) {
            case Instruction::UDiv:
            case Instruction::SDiv:
            case Instruction::URem:
            case Instruction::SRem:
                // Inactive lanes may hold a zero divisor; x86 would trap on it.
                // Not a constant one: no lane has zero, and keeping it constant
                // lets LLVM divide with shifts and multiplies instead of a
                // scalar idiv per lane.
                if (auto* c = dyn_cast<ConstantInt>(bin->getOperand(1)); c && !c->isZero()) break;
                rhs = b_.CreateSelect(mask, rhs, ConstantInt::get(rhs->getType(), 1));
                break;
            default:
                break;
        }
        Value* v = b_.CreateBinOp(bin->getOpcode(), lhs, rhs, bin->getName());
        if (auto* vi = dyn_cast<Instruction>(v)) vi->copyIRFlags(bin);
        return set(v);
    }
```

- `emit_uniform_division` ([14](../iteration-2/14-magic-number-division.md)) first handles the uniform non-constant divisor, with magic numbers computed at run time. Constant divisors arrive here.
- `vector(...)` of a constant produces a constant splat; without the `select`, the `sdiv` keeps a constant divisor and the following LLVM passes (InstCombine and the x86 instruction selection) reduce it.

### When it applies

Any kernel with `/` or `%` by a constant inside divergent code: block-stride loops (`for (e = threadIdx.x; e < N; e += blockDim.x)`), bodies of `if` that depend on `threadIdx`, loops with a per-lane exit. Where the mask is a constant with all lanes on, the `select` was already eliminated by LLVM and nothing changes.

### Limitation

With a constant divisor `-1` in an `sdiv`, an inactive lane with dividend `INT_MIN` would, by the semantics of the LLVM IR, perform a division with overflow (undefined behavior). InstCombine replaces `sdiv x, -1` with `0 - x`, so no `idiv` is executed, but the protection is no longer formal in that specific case.

## (b) Budget for the generic wide variants

### The problem

`cpusim-devc` generates, for each kernel, a **generic** module, compiled ahead of time and without knowing the block shape: the 32-lane variant, the instrumented one, the wide ones from 64 to 512 lanes, the ones that suspend at barriers and the cooperative ones ([02](../iteration-1/02-width-variants.md), [15](../iteration-2/15-smt-cooperative-variant.md)). At launch, the JIT recompiles the kernel specialized for the real block shape ([01](../iteration-1/01-specialized-jit.md)) and generates only the variants that shape can use.

The size of a variant grows with the width: an `i32` value of 512 lanes occupies 32 zmm registers, and each per-thread instruction becomes dozens of instructions. The code generation time grows faster than the size: the LLVM register allocator (the *greedy register allocator*) queries interferences between live ranges (`LiveIntervalUnion`) and decides where to insert spills (`SpillPlacement`), and with thousands of live values and high pressure this becomes superlinear.

The WMMA kernel of the test ([25](25-tensor-cores.md)) reached the extreme: the generic 512-lane variant had 40146 instructions, against at most 6560 in the matmul variants. A single instance of the kernel took 55 s to compile in the generic module. Specialized by the JIT, the same instance took 1.65 s for blocks of 128 threads, 3.66 s for 256 and 7.43 s for 512.

### The idea

In the generic module, the wide variants are a reserve: with the JIT on (the default), the one that runs is the specialized module. So, above a size limit, it is not worth generating the generic variant: it is dropped and the runtime never chooses it in the generic module. If the JIT is on, it generates the variant the real block needs, smaller (the thread indices and the block size become constants) and only at the applicable widths.

### The code

`src/compiler/main.cpp:383-412`

```cpp
        // Without a block shape (the module compiled ahead of time), a wide
        // variant past this many instructions is dropped: code generation
        // grows faster than linearly with size (register allocation), and the
        // JIT builds the variant the launch's block shape needs anyway.
        constexpr size_t kGenericBudget = 12000;
        bool dropped = false;
        auto build = [&](bool profile, unsigned lanes, bool suspending = false, bool cooperative = false) {
            ValueToValueMapTy vmap;
            Function* copy = CloneFunction(k.scalar, vmap);
            bool ok = cpusim::build_warp_entry(*m, k, *copy, profile, lanes, suspending, &why, cooperative);
            copy->eraseFromParent();
            if (ok && k.spec_block[0] == 0 && lanes > 32) {
                std::string name = "__cpusim_entry" + std::to_string(lanes) +
                                   (cooperative ? "c_" : suspending ? "s_" : "_") + k.name;
                if (Function* f = m->getFunction(name); f && f->getInstructionCount() > kGenericBudget) {
                    if (std::getenv("CPUSIM_DEVC_DEBUG"))
                        std::fprintf(stderr, "cpusim-devc: %s: %u instructions, left to the JIT\n", name.c_str(),
                                     f->getInstructionCount());
                    f->eraseFromParent();
                    dropped = true;
                }
            }
            return ok;
        };
        // build() for a variant the runtime may choose: false if dropped.
        auto kept = [&]() {
            bool d = dropped;
            dropped = false;
            return !d;
        };
```

- `k.spec_block[0] == 0` identifies the generic module (without a block shape); the JIT passes the shape and never drops anything.
- The 32-lane variant is never dropped (`lanes > 32`): it is the one that runs when no other exists.
- The size is measured in IR instructions, after the SIMT transformation and before code generation, which is the expensive part.
- `kept()` is consulted right after each `build` of a wide variant; only if the variant was kept does its bit enter `k.wide_variants`, the metadata the runtime reads to know which widths exist. A dropped variant simply does not appear to the runtime.
- `CPUSIM_DEVC_DEBUG` prints each dropped variant, with the number of instructions.

### The choice of the limit

With the limit at 16000, one instance of the WMMA kernel still took 4.7 s. With 12000, the compile time of the WMMA test dropped from 68.6 s to 35.6 s. The limit is above all the variants of the ordinary samples: matmul and heat2d (4.1 s and 5.1 s of compilation) did not lose any variant.

### When it applies (and when it does not)

Only to large kernels, whose generic wide variants exceed 12000 instructions: in general kernels with a lot of inlined code, such as those that use the tensor core API.

The trade-off: with `CPUSIM_JIT=0`, such a kernel only has the widths that fit in the budget (at minimum the 32-lane one), and the automatic choice works with fewer candidates. With the JIT on there is no difference in performance, because the generic module is not used for execution.

## (c) JIT cache key

### The problem

The JIT stores each specialized module on disk, in `~/.cache/cpusim/jit/<hash>-XxYxZ.so` ([01](../iteration-1/01-specialized-jit.md)). The hash covered the program bitcode, the version string and the ABI version between the generated code and the runtime. During development, the version string does not change at each build: a `cpusim-devc` rebuilt with a fix found in the cache the `.so` generated by the previous compiler, for the same bitcode, and reused it.

This actually happened: a measurement made right after fix (a) used modules generated before it, still with the `idiv`s, and the fix seemed to have no effect.

### The idea

Include in the key something that changes whenever the compiler binary changes: the size and the modification time (`st_mtim`, with nanoseconds) of the `cpusim-devc` executable. One `stat` per process is cheap; hashing the content of the binary would also cover a copy with preserved date, at the cost of reading the whole executable at the first launch.

### The code

`src/cudart/runtime.cpp:258-274`

```cpp
uint64_t bitcode_hash(const Module& m) {
    const auto* bc = reinterpret_cast<const uint8_t*>(m.header) + m.header->bitcode_offset;
    uint64_t h = 1469598103934665603ull;
    for (uint64_t i = 0; i < m.header->bitcode_size; ++i) h = (h ^ bc[i]) * 1099511628211ull;
    for (const char* v = CPUSIM_VERSION; *v; ++v) h = (h ^ uint8_t(*v)) * 1099511628211ull;
#ifdef CPUSIM_DEVC_PATH
    // The compiler that would build it: a rebuilt cpusim-devc (same version
    // string during development) must not reuse modules of the old one.
    static const uint64_t devc_id = []() -> uint64_t {
        struct stat st {};
        if (stat(CPUSIM_DEVC_PATH, &st) != 0) return uint64_t(0);
        return uint64_t(st.st_mtim.tv_sec) * 1000000007ull + uint64_t(st.st_mtim.tv_nsec) + uint64_t(st.st_size) * 31;
    }();
    h = (h ^ devc_id) * 1099511628211ull;
#endif
    return h ^ CPUSIM_KERNEL_ABI;
}
```

- The hash is 64-bit FNV-1a over the bitcode and the version, as before.
- `devc_id` is computed once per process (a local `static` variable, initialized on the first call) and combined into the hash as one more 64-bit "byte".
- If `stat` fails (the compiler is not at the configured path), `devc_id` is 0 and the key no longer distinguishes compiler builds.

### When it applies (and when it does not)

It only affects whoever rebuilds `cpusim-devc`. A normal installation keeps the same binary, and the cache remains valid between executions. Each new build of the compiler invalidates the whole cache (the old files stay in the directory, with no automatic cleanup).

The key does not cover the tensor core library or the Clang used to link the `.so`: the first is dynamically linked (a new version is loaded without recompiling the module), and the second only does the linking.

## Results

- **(a):** removed the 640 `idiv` from the hot function of `tensor_gemm`. The improvement in the sample was not measured in isolation: it came together with the other changes of the sequence described in [25](25-tensor-cores.md), from 0.19 to 0.49 TFLOPS in bfloat16.
- **(b):** compilation of the WMMA test from 68.6 s to 35.6 s; a WMMA kernel instance that took 55 s in the generic module no longer generates the wide variants; matmul (4.1 s) and heat2d (5.1 s) unchanged.
- **(c):** no performance number; it eliminates a source of wrong measurements during development.

## See also

- [00-basic-concepts.md](../00-basic-concepts.md): divergence, masks and the generic module.
- [01-specialized-jit.md](../iteration-1/01-specialized-jit.md): the JIT and the on-disk cache.
- [02-width-variants.md](../iteration-1/02-width-variants.md): the variants that the budget can drop.
- [14-magic-number-division.md](../iteration-2/14-magic-number-division.md): division by a uniform non-constant divisor.
- [21-compile-time.md](../iteration-2/21-compile-time.md): other compile time measurements.
- [25-tensor-cores.md](25-tensor-cores.md): where the three problems appeared.
