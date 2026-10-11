# 02. Width variants: 32, 64, 128... up to 512 lanes per call

> **In one sentence:** instead of generating a single piece of code that runs one warp (32 lanes) at a time, the compiler generates several versions of each kernel, each running a group of a different size (32, 64, 128, 256 or 512 lanes) and with different rules for barriers; larger groups give the CPU more independent work but use more registers, so the runtime picks the best one by measurement.

The project's general concepts (lane, group, frame, Gcyc) are in [00-basic-concepts.md](../00-basic-concepts.md).

## The problem

The most natural way to run CUDA on AVX-512 is: one call of the generated function executes **one warp**, 32 lanes, which fit in 2 zmm registers. Two problems appear.

### Problem 1: little independent work

The naive matmul loop (one thread per result element):

```cpp
float acc = 0;
for (int k = 0; k < N; ++k)
    acc += A[row * N + k] * B[k * N + col];
```

Each iteration does `acc = acc + a*b`, and the addition of the next iteration depends on the `acc` of the previous one: it is a **loop-carried dependency chain**, whose time per iteration is bounded by the latency of the addition, not by throughput.

On Zen 4, `vaddps` has a latency of 3 cycles and the FP units accept independent operations every cycle (512-bit instructions are executed as two 256-bit halves in the FP pipes). By Little's law, keeping the pipes busy requires at least latency × throughput independent operations in flight. With 32 lanes there are only 2 accumulator registers, that is, **2 chains**: after issuing the two additions, the scheduler has no other addition ready and the pipe sits idle until the result comes back.

The usual compiler solution would be to reassociate the sum (which LLVM does with `-ffast-math`/the `reassoc` flag): it would split `acc` into several partial accumulators (`acc0 += ...; acc1 += ...;` and at the end `acc0 + acc1`). Here this is **forbidden**: floating-point addition is not associative (`(a+b)+c` can differ from `a+(b+c)` in the last bit), and the project requires a result **bit-for-bit identical** to the GPU's. Each thread must sum in exactly the same order it would on the card.

The other source of instruction-level parallelism is **other CUDA threads**. Thread 0 and thread 40 each have their own `acc`, and the two chains are independent. If a call runs 128 lanes instead of 32, there are 8 accumulator registers, **8 independent chains**, enough to cover the addition latency without changing the order of operations inside each thread.

### Problem 2: expensive barriers

In kernels with `__syncthreads`, the 32-lane version has to suspend the warp at each barrier: save the state to the frame, return to the runtime, run the other warps of the block up to the same barrier and then resume each one. A block of 256 threads has 8 warps; each barrier costs 8 suspensions and 8 resumes.

## The idea

Increasing the width of the call multiplies the number of independent dependency chains per scalar instruction of the kernel: each `float` value now occupies `L/16` zmm registers, and each kernel operation becomes `L/16` independent instructions. This is ILP obtained from parallelism between threads, without reassociating anything inside a thread.

The limit is the register file: the number of zmm registers needed grows linearly with `L` times the number of simultaneously live values. When it exceeds 32, the register allocator generates spills (64-byte stores and reloads on the stack), which add load/store traffic and latency to the critical path.

For barriers, the argument is of a different nature: if **the whole block** runs in a single call, the SIMT linearization already executes all the threads of the block in lockstep, and when `__syncthreads` is reached all active lanes are at the same point of the program. The barrier generates no code at all.

Since the best point depends on the kernel, the compiler generates several versions and the runtime chooses ([03-automatic-width-selection.md](../iteration-2/03-automatic-width-selection.md)).

## How it works, step by step

### The variant families

Each kernel can have the following functions in the generated module. `L` is the width in lanes.

| Function name | When it exists | What it does at a barrier |
|---|---|---|
| `__cpusim_entry_<K>` | always | 32 lanes (one warp); suspends |
| `__cpusim_entry64_<K>`, `__cpusim_entry128_<K>` | kernels **without** a barrier | there is no barrier; 2 or 4 warps per call |
| `__cpusim_entry<L>_<K>`, L from 64 to 512 | kernels **with** a barrier | whole block in one call; the barrier becomes nothing |
| `__cpusim_entry64s_<K>`, `__cpusim_entry128s_<K>` | kernels with a barrier | groups of 2 or 4 warps that suspend |
| `__cpusim_entry<L>c_<K>`, L from 64 to 512 | kernels with a barrier | half a block on each SMT thread of a core |

`<K>` is the kernel name (the C++ "mangled" name).

### The naming convention

- **Number only** (`64`, `128`, `256`...): wide group that **does not suspend**. In a kernel without a barrier, it is several warps together. In a kernel with a barrier, it is the whole block.
- **Suffix `s`** (*suspending*): wide group that **suspends** at barriers, like the 32 one, but with fewer switches per block.
- **Suffix `c`** (*cooperative*): half of the block per call, the two halves running at the same time on the two SMT threads of the same core.

The same convention appears in the environment variable `CPUSIM_LANES` (for example `CPUSIM_LANES=128s`) and in the `CPUSIM_VERBOSE` messages.

### Example: a block of 8 threads, warps of 4

With warps of 4 lanes and a block of 8 threads with one barrier in the middle:

```text
"warp" variant (like the 32 one):
  call 1: warp A up to the barrier -> suspends
  call 2: warp B up to the barrier -> suspends
  call 3: warp A after the barrier -> done
  call 4: warp B after the barrier -> done
  (4 calls, 2 state saves, 2 resumes)

"whole block" variant (8 lanes):
  call 1: A and B together, pass through the barrier without stopping -> done
  (1 call, nothing saved)

"s" variant of 8 lanes on a block of 16 threads:
  like the "warp" one, but each call takes 2 warps: half the suspensions

"c" variant of 4 lanes on a block of 8 threads:
  SMT thread 1 runs A, SMT thread 2 runs B, at the same time;
  at the barrier, each one waits for the other to arrive (counters in memory)
```

### Why more lanes help: the chains

A `float` variable occupies `L / 16` zmm registers:

| Width | zmm per float variable | Independent `acc` chains |
|---|---|---|
| 32 | 2 | 2 |
| 64 | 4 | 4 |
| 128 | 8 | 8 |
| 256 | 16 | 16 |
| 512 | 32 | 32 (does not fit: there are only 32 zmm) |

In the matmul, going from 2 to 8 chains is enough to cover the 3-cycle latency of the addition.

### Why more lanes hurt: register pressure

Zen 4 has **32 zmm registers**. With 256 lanes, a single float variable occupies 16 of them, half of the total. A kernel with 3 or 4 vector values live at the same time (a 3D stencil that reads neighbors in x, y and z, for example) does not fit, and the allocator generates zmm spills to the stack. In fdtd3d (a 3D stencil) with 256 lanes, **79% of the cycles** were spent on spills.

For this reason no width is always the best, and the runtime decides by measurement.

### The context of each call

Every variant receives a single argument: a pointer to a **context** (`cpusim_warp_ctx`). It has a fixed 128-byte header (width, index of the first warp of the group, `blockIdx`, `blockDim`, pointers to parameters, shared memory and frame) followed by per-lane data, whose size depends on the width: the `threadIdx` of each lane and the mask of active lanes (lanes that correspond to threads that exist in the block). The CUDA warp semantics remain at 32: in a 128-lane call, lane 40 has `laneid` 8 (40 % 32), and `warpSize` is still 32.

### Which variants are generated

- **Generic module** (no JIT): always 32; for kernels without a barrier, 64 and 128; for kernels with a barrier, whole block from 64 to 512, suspending 64s and 128s and cooperative 64c to 512c. The generic module also has the instrumented variant (`__cpusim_prof_<K>`) for profiling.
- **JIT-specialized module** ([01-specialized-jit.md](01-specialized-jit.md)): only those that make sense for that block. For a 256-thread block with a barrier: 32, whole block 256 (and not 512), 64s, 128s and 128c (a block of 129 to 256 threads split into two halves of 128).

## The code

### Building the variants

`src/compiler/main.cpp:389-406`

```cpp
        auto build = [&](bool profile, unsigned lanes, bool suspending = false, bool cooperative = false) {
            ValueToValueMapTy vmap;
            Function* copy = CloneFunction(k.scalar, vmap);
            bool ok = cpusim::build_warp_entry(*m, k, *copy, profile, lanes, suspending, &why, cooperative);
            copy->eraseFromParent();
            return ok;
        };
```

Each variant starts from a copy of the "per-thread" kernel (the IR before vectorization). `build_warp_entry` is the SIMT transformation in [the simt.cpp file](../../../src/compiler/simt.cpp), which receives the width and the type (suspending or cooperative) and generates the vector function. The copy is erased afterwards because the transformation modifies it.

`src/compiler/main.cpp:439-461`

```cpp
        bool ok = build(false, 32) && (spec || build(true, 32));
        // Wider variants: with barriers, a whole block per call makes them free
        // (64..512 lanes); without, 2 or 4 warps per call give the core more
        // independent work. The runtime measures which one wins.
        for (unsigned i = 0; ok && i < (k.has_barriers ? 4u : 2u); ++i) {
            if (!wanted(64u << i, true)) continue;
            ok = build(false, 64u << i);
            if (ok && kept()) k.wide_variants |= 1u << (CPUSIM_META_WIDE_SHIFT + i);
        }
        // Groups of 2 and 4 warps that still suspend at barriers: fewer switches.
        for (unsigned i = 0; ok && k.has_barriers && i < 2; ++i) {
            if (!wanted(64u << i, false)) continue;
            ok = build(false, 64u << i, true);
            if (ok && kept()) k.wide_variants |= 1u << (CPUSIM_META_SUSP_SHIFT + i);
        }
        // Half a block per call, the two SMT siblings of a core together: for
        // blocks of L+1 to 2L threads.
        for (unsigned i = 0; ok && k.has_barriers && i < 4; ++i) {
            const unsigned lanes = 64u << i;
            if (spec && !(threads > lanes && threads <= 2 * lanes)) continue;
            ok = build(false, lanes, false, true);
            if (ok && kept()) k.wide_variants |= 1u << (CPUSIM_META_COOP_SHIFT + i);
        }
```

Details:

- Line 439: always generates the 32-lane one; in generic mode (`!spec`), also the instrumented one (`build(true, 32)`).
- `64u << i` runs through 64, 128, 256, 512 for `i` = 0, 1, 2, 3.
- First loop: variants that do not suspend. With a barrier it goes up to 512 (`i < 4`); without a barrier, only 64 and 128 (`i < 2`). `wanted` (see [01-specialized-jit.md](01-specialized-jit.md)) filters by block size in JIT mode.
- Second loop: suspending 64s and 128s, only for kernels with a barrier.
- Third loop: cooperative, for blocks that have between `L+1` and `2L` threads.
- Each generated variant sets a bit in `k.wide_variants`. These bits go into a kernel metadata table (`__cpusim_meta_<name>`) that the runtime reads to know what exists. `kept()` is false when the generic variant exceeded the instruction budget and was dropped in favor of the JIT ([26](../iteration-3/26-compiler-and-runtime-changes.md)).

### The name of each variant

`src/compiler/simt.cpp:1899-1905`

```cpp
    w_ = Function::Create(FunctionType::get(i32, {ptr}, false), GlobalValue::ExternalLinkage,
                          (profile_ ? std::string("__cpusim_prof_")
                           : W == CPUSIM_WARP_SIZE ? std::string("__cpusim_entry_")
                                                   : "__cpusim_entry" + std::to_string(W) +
                                                         (coop_ ? "c_" : block_wide_ ? "_" : "s_")) +
                              k_.name,
                          m_);
```

This is where the convention is born: 32 lanes becomes `__cpusim_entry_`, and the other widths get the number and the suffix `c_`, `_` or `s_`.

### The contract with the runtime

`src/device/kernel_abi.h:30-48`

```cpp
// Besides the 32-lane entry, a module has "block-wide" variants
//     uint32_t __cpusim_entry<L>_<mangled name>(const cpusim_warp_ctx*);
// for L = 64, 128, 256, 512 lanes (listed in cpusim_kernel_meta.flags). When a
// whole block fits in one call, its threads run in lockstep and __syncthreads
// costs nothing: no frame, no resume. The runtime picks the smallest variant
// that covers the block for kernels with barriers.
//
// Kernels with barriers also get "suspending" variants of 64 and 128 lanes,
//     uint32_t __cpusim_entry<L>s_<mangled name>(const cpusim_warp_ctx*);
// which behave like the 32-lane entry (save state, return at a barrier) for
// groups of 2 or 4 warps: fewer switches per barrier. frame_bytes covers them.
//
// And "cooperative" variants of 64, 128, 256 and 512 lanes,
//     uint32_t __cpusim_entry<L>c_<mangled name>(const cpusim_warp_ctx*);
// for blocks of L+1 to 2L threads: the two SMT siblings of a core run the two
// halves of one block at the same time (warp_index 0 and L/32), sharing its
// shared memory. Each __syncthreads is inline code (no call: a call would
// spill every vector register): bump this half's epoch, wait until the other
// half's epoch is at least as high. No frames, no suspending.
```

This comment is the "contract" between the generated code and the runtime. One remark: the sentence "the runtime picks the smallest variant that covers the block" means that, among the block-wide ones, only the smallest one that covers the block enters the contest; the final choice among it, the suspending ones, the cooperative ones and the 32 one is made by measurement ([03-automatic-width-selection.md](../iteration-2/03-automatic-width-selection.md)). The comment also mentions 64 to 512 for the block-wide ones, which is the case for kernels with a barrier; kernels without a barrier have only 64 and 128.

`src/device/kernel_abi.h:55-81`

```cpp
// One call's context: this 128-byte header, followed by the per-lane data
// sized for the entry's width (cpusim_ctx_size):
//     uint32_t tid_x[lanes], tid_y[lanes], tid_z[lanes];   // thread index per lane
//     uint32_t lane_mask[lanes / 32];                        // bit set: real thread
// Compact on purpose: a block's contexts must stay in L1 while its warps take
// turns at barriers.
typedef struct __attribute__((aligned(64))) cpusim_warp_ctx {
    uint32_t lanes;         // lanes in this call (the entry's width)
    uint32_t warp_index;    // first warp of this group inside the block
    uint32_t block_idx[3];
    uint32_t block_dim[3];
    uint32_t grid_dim[3];
    // ...
    uint32_t barrier_pass;
    const uint8_t* params;  // kernel arguments, laid out as in the module manifest
    uint8_t* shared;        // the block's shared memory
    uint64_t* counters;     // CPUSIM_COUNTERS slots, only used by the __cpusim_prof_ entry
    uint8_t* frame;         // this group's saved state across barriers (64-byte aligned)
    // ...
    uint64_t* coop_epoch;
    const uint64_t* coop_other;
} cpusim_warp_ctx;
```

The same header serves all widths; the `lanes` field says how many lanes follow. `warp_index` is the first warp of the group: in a 128s call that runs warps 4 to 7 of a block, it is 4. `coop_epoch` and `coop_other` are only used by the cooperative variants.

### The metadata bits

`src/device/kernel_abi.h:115-119`

```cpp
#define CPUSIM_META_HAS_BARRIERS 1u
#define CPUSIM_META_WIDE_SHIFT 8   // bit (8 + i): block-wide variant with 64 << i lanes exists
#define CPUSIM_META_SUSP_SHIFT 16  // bit (16 + i): suspending variant __cpusim_entry<64 << i>s_ exists
#define CPUSIM_META_COOP_SHIFT 24  // bit (24 + i): cooperative variant __cpusim_entry<64 << i>c_ exists
#define CPUSIM_META_BLOCK_FN 2u    // __cpusim_block_<name> exists (see below)
```

The field is a 32-bit integer: bit 0 says whether the kernel has a barrier; bits 8 to 11, which non-suspending variants exist; bits 16 and 17, which suspending ones; bits 24 to 27, which cooperative ones. Example: a kernel with a barrier, compiled by the JIT for a 256 block, would have bits 0, 10 (256 = 64 << 2), 16, 17 and 25 (128c = 64 << 1) set.

### On the runtime side: the `s` and `c` marks

`src/device/device_core.cpp:270-275`

```cpp
// Widths are lane counts; kSuspend marks the 64/128-lane variants that still
// suspend at barriers (several groups per block, like 32), kCoop the
// cooperative ones (half a block on each SMT sibling of a core).
constexpr uint32_t kSuspend = 0x10000;
constexpr uint32_t kCoop = 0x20000;
const char* width_suffix(uint32_t w) { return w & kSuspend ? "s" : w & kCoop ? "c" : ""; }
```

In the runtime, a "width" is a number whose low 16 bits are the lanes and a higher bit marks `s` or `c`. Thus `128 | kSuspend` and `128` are different candidates, and `width_suffix` produces the text `128s` in the messages.

## When it applies (and when it does not)

- **Kernels without a barrier** receive 32, 64 and 128. There is no 256 or 512: without a barrier, there is no "the barrier becomes nothing" gain, and register pressure grows linearly with the width. In the JIT, widths larger than the block are not generated.
- **Kernels with a barrier** receive 32, the block-wide one (only if the block has up to 512 threads), 64s/128s (only if the block is larger than the width) and the cooperative ones (only if the block has between `L+1` and `2L` threads, and only if the executor has pairs of SMT threads).
- **Blocks of 1024 threads** with a barrier have no block-wide variant (the maximum is 512 lanes, `CPUSIM_MAX_LANES`); they use 32, the suspending ones or the cooperative 512c.
- **Heavy divergence** can favor small groups: in the SIMT linearization, a branch is only skipped when the mask of lanes that take it is entirely zero. With 32 lanes this happens more often; with 256, a single active lane in the branch is enough for the whole group to execute its instructions (with the remaining lanes masked).
- **Many live variables** (3D stencils, long kernels) disfavor large widths because of spills.

## Results

Widths typically chosen by the automatic measurement:

| Kernel | Chosen width |
|---|---|
| matmul naive | 128 |
| matmul tiled | 256 (whole block) |
| iso2dfd | 128 |
| fdtd3d | 128c |
| heat2d (HeCBench suite) | 32 |

Tiled matmul (16x16 block, with a barrier), in billions of cycles (lower is better):

| Variant | Gcyc |
|---|---|
| 32 | 30.5 |
| 64s | 24.2 |
| 128s | 21.8 |
| 128c | 21.9 |
| 256 (whole block) | 14.35 |

Naive matmul (no barrier):

| Variant | Gcyc |
|---|---|
| 32 | 21.5 |
| 64 | 14.9 |
| 128 | 13.96 |

In the naive one, going from 2 to 8 addition chains removed 35% of the cycles. In the tiled one, the whole-block variant beats the suspending ones because it eliminates all the suspensions at barriers. In the opposite direction, fdtd3d with 256 lanes spent 79% of the cycles on spills, and the best choice there was the cooperative 128.

## Limitations and trade-offs

- **Compile time and module size:** each variant is a complete compilation of the kernel. The generic module of a kernel with a barrier has more than ten functions. The JIT reduces this by generating only the applicable ones ([21-compile-time.md](../iteration-2/21-compile-time.md)).
- **No width always wins:** hence the measurement ([03-automatic-width-selection.md](../iteration-2/03-automatic-width-selection.md)), which has its own cost.
- **Cooperative variants** depend on two SMT threads of the same core being free at the same time and wait for each other at each barrier; details in [15-smt-cooperative-variant.md](../iteration-2/15-smt-cooperative-variant.md).

## See also

- [00-basic-concepts.md](../00-basic-concepts.md): lanes, registers, spill, SMT.
- [01-specialized-jit.md](01-specialized-jit.md): how the block shape reduces the generated variants.
- [03-automatic-width-selection.md](../iteration-2/03-automatic-width-selection.md): how the runtime chooses.
- [09-state-at-barriers.md](09-state-at-barriers.md) and [10-barrier-pass.md](../iteration-2/10-barrier-pass.md): the cost of suspending in the 32 and `s` variants.
- [15-smt-cooperative-variant.md](../iteration-2/15-smt-cooperative-variant.md): the `c` variants in detail.
- [16-splitting-blocks-into-warps.md](../iteration-2/16-splitting-blocks-into-warps.md): how the groups of a block are distributed.
