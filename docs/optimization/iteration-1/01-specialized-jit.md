# 01. Launch-time JIT specialized by block shape

> **In one sentence:** the first time a kernel is launched with a given block size, the runtime recompiles the kernel knowing that size; as a result `blockDim` becomes a fixed number and `threadIdx` becomes a table of known numbers, and the compiler can resolve at compile time questions that used to be asked at every memory access.

The project's general concepts (lane, group, uniform, variants) are in [00-basic-concepts.md](../00-basic-concepts.md).

## The problem

When `cucc` compiles a CUDA program, it does not know what block size the kernel will be launched with. That information only appears at run time, at the line `kernel<<<grid, block>>>(...)`, and the same kernel can even be launched with different sizes.

Without the block size, the generated code has to:

1. **read `threadIdx` from memory.** For each group of lanes the runtime fills a table with the `threadIdx.x`, `y` and `z` of each lane, and the code loads that table into a vector with a plain `load`. To LLVM this vector is opaque: no analysis (SCEV, `computeKnownBits`, InstCombine) can deduce a relation between lanes, so every index derived from it is treated as arbitrary.
2. **discover the pattern of each access at run time.** An access such as `B[k * N + threadIdx.x]` is contiguous (the lanes read neighbors) only if `threadIdx.x` is 0, 1, 2, 3... in sequence. Since this cannot be proven at compile time, the generated code tests at run time, on every access, whether the 16 addresses of the lane block are consecutive or all equal, and chooses between a contiguous load, a broadcast or a gather (see [06-dynamic-coalescing.md](06-dynamic-coalescing.md)).
3. **keep one pointer per group of 16 lanes.** Addresses that would be "base + constant" become a scalar pointer computed and kept live in a GPR for each block of 16 lanes. In the wide variants (256 lanes = 16 blocks of 16), this exceeds the 16 GPRs of x86-64 and generates spills to the stack.

Concrete example: in the tiled matmul (shared memory, 16x16 block), the inner loop reads `As[ty][k]` and `Bs[k][tx]`. With a 16x16 block, `As[ty][k]` is uniform across the 16 lanes of a block row and `Bs[k][tx]` is a contiguous 64-byte row. Without the block size, neither property can be proven by the compiler.

## The idea

At launch time the block size is known. The idea is to specialize the kernel by that value (like an interprocedural constant propagation done late, at run time): recompile the bitcode with the block shape fixed and store the resulting object in a content-indexed disk cache. The compilation cost is paid once per shape; the generic code remains available as a fallback. With the size known:

- `blockDim.x`, `blockDim.y`, `blockDim.z` become constants;
- `threadIdx` becomes a **constant vector** (for example `<0, 1, 2, ..., 15, 0, 1, ...>`) or a constant added to a single uniform number;
- the mask of "lanes that are real threads" becomes a constant when the block is a multiple of the group width.

With these constants, LLVM's InstCombine, SCEV and constant folding evaluate the pattern tests at compile time, and the accesses become contiguous loads/stores or broadcasts with an immediate displacement.

## How it works, step by step

### 1. The bitcode ships inside the executable

In real CUDA, the fatbin carries the code in two forms: SASS (binary for specific architectures) and **PTX**, a virtual ISA that the driver JIT-compiles for the GPU present. The project mimics this: `cucc` embeds in the executable (in the "fatbin") both the generic module (an already compiled `.so`) and the kernel's **per-thread bitcode**, which is the LLVM IR saved in binary form, before any vectorization. The fatbin header has the fields `bitcode_offset` and `bitcode_size` so that the runtime can find it.

### 2. On the first launch of each shape

When the program launches a kernel, `libcpusim_cudart` (the runtime that replaces the CUDA runtime) looks at the block shape, for example 16x16x1. If code for that shape does not exist yet:

```text
1. computes a hash of the bitcode (+ project version + ABI version)
2. looks for ~/.cache/cpusim/jit/<hash>-16x16x1.so
3. if it does not exist:
     writes the bitcode to a temporary file
     runs   cpusim-devc tmp.bc -o tmp.o --specialize 16,16,1
     runs   clang -shared -o tmp.so tmp.o ...
     renames tmp.so to the final name
4. loads the .so and looks up the kernel function
```

A typical compilation takes 0.5 s. Since the file stays in the disk cache, later runs of the same program do not recompile. If the same kernel is launched with another shape (for example 32x8x1), another file is generated.

### 3. Inside the compiler: `threadIdx` becomes a constant

`cpusim-devc` receives `--specialize X,Y,Z` and, when generating the vector code, replaces the reads of `threadIdx` and `blockDim` with computed values. There are three cases.

**Case A: the group covers the whole block.** Example: a 4x2 block (8 threads) running in a group of 8 lanes:

```text
lane:        0  1  2  3  4  5  6  7
threadIdx.x: 0  1  2  3  0  1  2  3      (lane % 4)
threadIdx.y: 0  0  0  0  1  1  1  1      (lane / 4)
```

All these numbers are known at compile time. If the block has fewer threads than lanes (for example 6 threads in a group of 8), lanes 6 and 7 get mask 0, which is also constant.

**Case B: several groups per block, and the width is a multiple of `blockDim.x`.** A 4x4 block (16 threads) in groups of 8 lanes. Group 0 has rows y=0 and y=1; group 1 has y=2 and y=3:

```text
                  lane: 0 1 2 3 4 5 6 7
threadIdx.x (always):   0 1 2 3 0 1 2 3          constant
threadIdx.y in group 0: 0 0 0 0 1 1 1 1  = 0 + <0,0,0,0,1,1,1,1>
threadIdx.y in group 1: 2 2 2 2 3 3 3 3  = 2 + <0,0,0,0,1,1,1,1>
```

`threadIdx.x` is constant; `threadIdx.y` is "a uniform number (the first row of the group) plus a constant vector". The uniform number is computed from the warp index, a single integer.

**Case C: none of the above.** It computes `t = base + lane` and from it `x = t % bx`, `y = (t / bx) % by`, `z = t / (bx*by)`, still with constants in the divisors, which LLVM turns into cheap arithmetic.

### 4. The effect on the final code

With `threadIdx` known, the access `Bs[k][tx]` becomes "base address + k*64 + constant", the same for all blocks of 16 lanes; `As[ty][k]` becomes "a single float replicated across 16 lanes" (a *broadcast*). The inner loop of the tiled matmul became a sequence of instructions such as:

```text
vmulps  -0x3c0(%rcx){1to16}, %zmmB, %zmmT     ; reads 1 float from As, replicates to 16, multiplies
vaddps  %zmmT, %zmmAcc, %zmmAcc               ; accumulates
```

The `{1to16}` is the AVX-512 embedded broadcast (a 4-byte memory operand replicated across the 16 lanes, with no separate `vbroadcastss` instruction), and the displacement `-0x3c0` is a constant computed at compile time. The loop ended up with 16 accumulators (one per block of 16 lanes of the 256-lane group), with no pattern test on the path.

### 5. Only the variants that can be used

The generic module contains many versions of each kernel (32 lanes, 64, 128, ..., see [02-width-variants.md](02-width-variants.md)), because it does not know which one will fit. In specialized mode, knowing the block size, the compiler generates only the ones the runtime can actually choose for that block. It also does not generate the instrumented profiling variant: when profiling is on, the runtime uses the generic module.

## The code

### The JIT switch

`src/cudart/runtime.cpp:215-232`

```cpp
// ---- JIT: recompiling the device bitcode for the launch's block shape
//
// With blockDim known, threadIdx becomes a constant (or constant + uniform)
// vector and the compiler resolves address patterns statically. The result is
// cached on disk by content, so only the first run of a program pays for it.
// CPUSIM_JIT=0 disables it; CPUSIM_VERBOSE=1 reports compilations.

bool jit_enabled() {
#if defined(CPUSIM_DEVC_PATH) && defined(CPUSIM_CLANG_PATH)
    static const bool on = [] {
        const char* v = std::getenv("CPUSIM_JIT");
        return !(v && std::strcmp(v, "0") == 0);
    }();
    return on;
#else
    return false;
#endif
}
```

The JIT is on by default. The environment variable `CPUSIM_JIT=0` turns it off. If the runtime was compiled without knowing where `cpusim-devc` and `clang` are, the JIT is not compiled in and everything runs on the generic code.

### Compiling (or finding it in the cache)

`src/cudart/runtime.cpp:276-305`

```cpp
// Path of the specialized shared object, compiling it if it is not cached.
bool jit_object(const Module& m, const std::array<uint32_t, 3>& block, std::string* path) {
#if defined(CPUSIM_DEVC_PATH) && defined(CPUSIM_CLANG_PATH)
    const auto* base = reinterpret_cast<const uint8_t*>(m.header);
    const uint8_t* bc = base + m.header->bitcode_offset;
    const uint64_t n = m.header->bitcode_size;
    const uint64_t h = bitcode_hash(m);

    char name[96];
    std::snprintf(name, sizeof(name), "/%016llx-%ux%ux%u", (unsigned long long)h, block[0], block[1], block[2]);
    std::string stem = jit_cache_dir() + name;
    *path = stem + ".so";
    if (access(path->c_str(), R_OK) == 0) return true;

    std::string tmp = stem + "." + std::to_string(getpid());
    {
        std::ofstream f(tmp + ".bc", std::ios::binary);
        f.write(reinterpret_cast<const char*>(bc), std::streamsize(n));
        if (!f) return false;
    }
    char spec[48];
    std::snprintf(spec, sizeof(spec), "%u,%u,%u", block[0], block[1], block[2]);
    auto start = std::chrono::steady_clock::now();
    bool ok = run_tool({CPUSIM_DEVC_PATH, tmp + ".bc", "-o", tmp + ".o", "--manifest", tmp + ".manifest",
                        "--specialize", spec}) &&
              run_tool({CPUSIM_CLANG_PATH, "-shared", "-nostdlib", "-o", tmp + ".so", tmp + ".o", "-Wl,--as-needed",
                        "-lmvec", "-lm", "-L" CPUSIM_LIB_DIR_PATH, "-lcpusim_tensor",
                        "-Wl,-rpath," CPUSIM_LIB_DIR_PATH}) &&
              std::rename((tmp + ".so").c_str(), path->c_str()) == 0;
    for (const char* ext : {".bc", ".o", ".manifest", ".so"}) std::remove((tmp + ext).c_str());
    // ...
```

Step by step:

- `bitcode_hash` summarizes the bitcode into a 64-bit number, together with the project version, the ABI and the size and modification date of the `cpusim-devc` binary, so that a new compiler does not reuse files generated by the old one ([26](../iteration-3/26-compiler-and-runtime-changes.md)).
- The file name is `<hash>-XxYxZ.so`, inside `~/.cache/cpusim/jit` (or `$XDG_CACHE_HOME/cpusim/jit`, if that variable exists; the function `jit_cache_dir` decides).
- If the file already exists (`access(...) == 0`), there is nothing to do.
- Otherwise, it writes the bitcode to a temporary file with the process number in the name (so that two programs running at the same time do not interfere with each other), calls `cpusim-devc` with `--specialize X,Y,Z`, then `clang` to turn the object into a library, and renames the result. The `rename` is atomic: either another process sees the complete file, or it sees nothing.
- With `CPUSIM_VERBOSE=1`, it prints how long it took (lines 294-297).

### Using the result at launch

`src/cudart/runtime.cpp:416-432`

```cpp
// The device function to launch for this block shape, or 0 to use the generic one.
uint64_t jit_function(Module& m, const KernelDesc& k, const std::array<uint32_t, 3>& block, cpusim::Device* d) {
    if (!jit_enabled() || !m.header->bitcode_size) return 0;
    JitModule& j = m.jit[block];
    if (!j.tried) {
        j.tried = true;
        std::string path;
        if (jit_object(m, block, &path)) {
            std::ifstream f(path, std::ios::binary);
            std::string bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
            j.ok = !bytes.empty() && d->load_module(bytes.data(), bytes.size(), &j.handle) == Error::Success;
        }
        if (!j.ok && std::getenv("CPUSIM_VERBOSE"))
            std::fprintf(stderr, "cpusim: JIT unavailable, using the generic kernels\n");
    }
    if (!j.ok) return 0;
    auto it = j.functions.find(k.name);
    // ...
```

There is one record per block shape (`m.jit[block]`). The attempt is made only once (`j.tried`): if it fails, the program continues with the generic module, without trying again at every launch. Returning 0 means "use the generic one".

`src/cudart/runtime.cpp:975-979`

```cpp
    // Instrumented launches run the generic module (the JIT one has no
    // instrumented entries).
    bool specialized = false;
    if (!profiler().enabled)
        if (uint64_t jf = jit_function(m, k, {block.x, block.y, block.z}, d)) function = jf, specialized = true;
```

On the launch path: if profiling is off, it tries the JIT; if that returns a function, it replaces the generic one.

### `blockDim` becomes a constant

`src/compiler/simt.cpp:1949-1970`

```cpp
    if (k_.spec_block[0]) {
        special_[(int)SpecialReg::NtidX] = b_.getInt32(k_.spec_block[0]);
        special_[(int)SpecialReg::NtidY] = b_.getInt32(k_.spec_block[1]);
        special_[(int)SpecialReg::NtidZ] = b_.getInt32(k_.spec_block[2]);
```

`Ntid` is the internal name for `blockDim`. In specialized mode, instead of a load from the context, each component is an `i32` constant with the value passed in `--specialize`.

### `threadIdx` becomes a constant vector

`src/compiler/simt.cpp:1529-1566`

```cpp
// threadIdx for a known block shape. Lane l of the group starting at thread
// `base` (= warp_index * 32) is thread base + l. When the group covers the
// whole block, base is 0 and every index is a constant vector; when the width
// is a multiple of blockDim.x and groups stay inside one z-plane, x is a
// constant vector and y, z are a constant vector plus a uniform value.
void WarpBuilder::emit_specialized_tid() {
    const uint32_t bx = k_.spec_block[0], by = k_.spec_block[1], bz = k_.spec_block[2];
    const uint32_t threads = bx * by * bz;
    // ...
    if (threads <= W) {
        special_[(int)SpecialReg::TidX] = lanes([&](unsigned l) { return l < threads ? l % bx : 0; });
        special_[(int)SpecialReg::TidY] = lanes([&](unsigned l) { return l < threads ? l / bx % by : 0; });
        special_[(int)SpecialReg::TidZ] = lanes([&](unsigned l) { return l < threads ? l / (bx * by) : 0; });
        SmallVector<Constant*, 64> m;
        for (unsigned l = 0; l < W; ++l) m.push_back(b_.getInt1(l < threads));
        spec_lane_mask_ = ConstantVector::get(m);
        return;
    }
    // base = warp_index * 32, a multiple of W here (groups are W lanes).
    Value* warp = b_.CreateAlignedLoad(i32, b_.CreateConstInBoundsGEP1_64(b_.getInt8Ty(), ctx_arg_, CPUSIM_CTX_WARP_INDEX),
                                       Align(4), "warp");
    // ...
    Value* base = b_.CreateMul(warp, b_.getInt32(CPUSIM_WARP_SIZE), "base", true, true);
    // ...
    if (W % bx == 0 && (bx * by) % W == 0) {
        Value* q = b_.CreateUDiv(base, b_.getInt32(bx));  // first row of the group
        special_[(int)SpecialReg::TidX] = lanes([&](unsigned l) { return l % bx; });
        special_[(int)SpecialReg::TidY] =
            b_.CreateAdd(b_.CreateVectorSplat(W, b_.CreateURem(q, b_.getInt32(by))), lanes([&](unsigned l) { return l / bx; }),
                         "tid.y", true, true);
        special_[(int)SpecialReg::TidZ] = b_.CreateVectorSplat(W, b_.CreateUDiv(q, b_.getInt32(by)), "tid.z");
```

`W` is the group width (number of lanes). The helper function `lanes(f)` builds a constant vector by applying `f` to each lane number.

- **Case A** (`threads <= W`): the group covers the whole block. `x = l % bx`, `y = l / bx % by`, `z = l / (bx*by)`, all constant. Lanes beyond the block get 0 and are left out of the constant mask `spec_lane_mask_`.
- **Case B** (`W % bx == 0 && (bx*by) % W == 0`): the only thing read at run time is the warp index (an integer). From it comes `q`, the first row of the group. `x` is constant; `y` is `q % by` replicated (uniform) plus a constant vector; `z` is uniform. The `MD_range` annotation (omitted above) tells LLVM that the warp index is small, which helps prove that the arithmetic does not overflow.
- Case C (lines 1559-1565) and the mask (lines 1567-1572) complete the function: the mask is "all lanes" when the block is a multiple of `W`.

### Only the variants the runtime can choose

`src/compiler/main.cpp:429-438`

```cpp
        // With a known block shape (JIT at launch) only the variants the
        // runtime can pick are built, and no instrumented one (profiling runs
        // the generic module).
        const bool spec = k.spec_block[0] != 0;
        const uint64_t threads = uint64_t(k.spec_block[0]) * k.spec_block[1] * k.spec_block[2];
        auto wanted = [&](unsigned lanes, bool block_wide) {
            if (!spec) return true;
            if (block_wide) return k.has_barriers ? threads <= lanes && threads > lanes / 2 : lanes <= threads;
            return lanes < threads;  // suspending variants: several groups per block
        };
```

In generic mode (`!spec`) everything is generated. In specialized mode, for a kernel with barriers only the block-wide variant that covers the block "snugly" is generated (for example 256 lanes for a 256-thread block, and not 512); for a kernel without barriers, only widths that do not exceed the block size. This reduces the JIT compilation time. The details of each variant are in [02-width-variants.md](02-width-variants.md).

## When it applies (and when it does not)

- **It applies** to every normal launch when the runtime has the path to `cpusim-devc` and `clang` and the executable carries the embedded bitcode.
- **It does not apply** when:
  - `CPUSIM_JIT=0` is set;
  - profiling is on (the specialized module has no instrumented variant);
  - the executable was generated by an old `cucc`, without bitcode (`bitcode_size == 0`);
  - compilation fails for any reason. In that case a message appears with `CPUSIM_VERBOSE=1` and the program continues on the generic code, which is slower but gives the same result.
- **The cost** is paid once per block shape and per program version: about 0.5 s on the first run. Programs that launch a kernel with many different block shapes pay one compilation per shape.
- **The gain** is larger when the kernel indexes memory with `threadIdx` (almost all of them) and when wide variants are used, in which per-block pointers for each 16 lanes would weigh on the registers.

## Results

It was the largest single gain recorded in the project. Measured right when it landed, before the address decomposition improvements that came later ([05-address-decomposition.md](05-address-decomposition.md)):

| Kernel | Without JIT | With JIT |
|---|---|---|
| matmul naive | ~313 ms | 280 ms |
| matmul tiled | ~600 ms | ~390 ms |

Besides the time, the code changed in nature: the inner loop of the tiled matmul became a sequence of `vmulps -0x3c0(%rcx){1to16}, ...` with 16 accumulators, with no pattern tests. The JIT was also what made the block-wide variant useful: without specialization it was worse than the 32-lane one, for lack of registers ([22-what-did-not-work.md](../iteration-2/22-what-did-not-work.md)).

## Limitations and trade-offs

- **First-run time:** about 0.5 s per block shape. For a program that runs a few milliseconds, this dominates; on later runs the disk cache eliminates the cost.
- **Disk space:** one `.so` per (program, block shape) in `~/.cache/cpusim/jit`. There is no automatic cleanup.
- **Tool dependency:** the runtime must find `cpusim-devc` and `clang` on the machine where the program runs.
- **Fixed shape, not grid:** only the block is specialized. The grid size (`gridDim`) and the kernel parameters are still read at run time.

## See also

- [00-basic-concepts.md](../00-basic-concepts.md): terms used here.
- [02-width-variants.md](02-width-variants.md): the widths generated for each kernel.
- [03-automatic-width-selection.md](../iteration-2/03-automatic-width-selection.md): how the runtime chooses among them.
- [05-address-decomposition.md](05-address-decomposition.md) and [06-dynamic-coalescing.md](06-dynamic-coalescing.md): the pattern tests that the JIT allows to be resolved at compile time.
- [12-static-access-analysis.md](../iteration-2/12-static-access-analysis.md): access analysis that takes advantage of the known `threadIdx`.
- [21-compile-time.md](../iteration-2/21-compile-time.md): the cost of compiling.
