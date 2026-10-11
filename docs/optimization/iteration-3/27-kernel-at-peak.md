# 27. A kernel written for the CPU, close to the peak

> **In one sentence:** a CUDA GEMM written for the execution model of `cucc` (one-warp blocks, no barriers, accumulators sized for the 32 zmm registers, data packed for L1 and L2) reaches 94% of the peak of one core in fp32 and 93% in bfloat16; for bfloat16 to get there, `mma_sync` stopped calling the library and came to be emitted inline with the accumulators in registers (`__cpusim_mmav_bf16_f32`, `emit_mmav`); with all cores, 1.64 TFLOPS in fp32 and 3.17 TFLOPS in bfloat16.

The general concepts (lanes, variants, per-thread arrays, SMT) are in [00-basic-concepts.md](../00-basic-concepts.md). The tensor core API, the distribution of the elements in the fragments and the pointer path are in [25](25-tensor-cores.md).

## The problem

The previous documents optimize the compiler for CUDA code written for the GPU. A kernel written for the GPU uses blocks of hundreds of threads, shared memory and `__syncthreads`, and counts on thousands of registers per SM. On the CPU this has costs that the compiler only reduces up to a point: each barrier suspends the warp and saves state ([09](../iteration-1/09-state-at-barriers.md), [10](../iteration-2/10-barrier-pass.md)), shared memory is one more copy that goes through the caches, and live values beyond 32 zmm become spills. `samples/tensor_gemm.cu` ([25](25-tensor-cores.md)), written in the GPU style, stayed at 0.62 TFLOPS in bfloat16 at 2048^3.

The question of this work: writing the CUDA kernel for our architecture (and not for the GPU), how close to the theoretical peak of Zen 4 can we get, in fp32 and in bfloat16? Optimizing the CUDA code, and not only the compiler, makes it possible to avoid context switches, use the registers on purpose and have no spills. The result is `samples/cpu_gemm.cu`, which remains valid CUDA and also compiles with `nvcc`.

## The theoretical peak

Zen 4 has two 256-bit FMA units and executes a zmm instruction as two 256-bit halves; the throughput is **one zmm FMA per cycle** (16 lanes x 2 FLOPs = 32 FLOPs in fp32) and **one zmm `vdpbf16ps` per cycle** (16 lanes x 2 products x 2 FLOPs = 64 FLOPs). To reach this throughput, there must be enough independent chains to cover the latency of the instruction.

The peak was measured in a microbenchmark with 12 independent accumulators, compiled with Clang:

| Configuration | fp32 | bf16 |
|---|---|---|
| 1 thread | 154 GFLOPS | 307 GFLOPS |
| 14 threads (one per core) | ~1.96 TFLOPS | ~4.0 TFLOPS |
| 28 threads (SMT) | no gain over 14 | no gain over 14 |
| Sustained for ~5 s, 4.5 GHz | 1992 GFLOPS | 3956 GFLOPS |

With one thread per core, the independent chains already occupy the FMA units; the second SMT thread shares the same units and adds no throughput.

The first version of the microbenchmark was wrong. Compiled with GCC, it left the array of `__m512` accumulators on the stack (42 instructions per iteration instead of ~13): an `asm` compiler barrier with a `memory` clobber forced the array to be written and read back, and a run-time branch on the operation type (`kind`) prevented the indices from becoming constants. The correct version uses templates (the operation type is a compile-time parameter) and Clang.

**Temperature.** The laptop heated up over the course of the session, and the measured peak varied from 1800 to 1990 GFLOPS in fp32 and from 3200 to 3950 GFLOPS in bf16. For this reason the all-cores efficiency was always computed against a peak measured right before the GEMM, and not against the table above.

## The idea

1. **A block is a warp.** With 32 threads and no barriers or shared memory, the 32-lane variant executes the block from start to finish, without suspending and without context switches ([02](../iteration-1/02-width-variants.md)).
2. **The accumulators fit in registers.** With 32 lanes, one `float` per thread is a `<32 x float>` vector, 2 zmm. The number of accumulators per thread is chosen so that accumulators, operands and broadcasts fit in 32 zmm.
3. **The operand common to the warp is uniform.** A value equal in all lanes is a scalar load and a broadcast (which on x86 can be a `{1to16}` memory operand), and not a per-lane vector load.
4. **The data is packed** by copy kernels so that each step of `k` reads contiguous addresses, and the loop over `k` is split into passes that keep the panels in L1 and L2, as in BLIS.
5. **In bfloat16, the compiler also changes:** the pointer `mma_sync` ([25](25-tensor-cores.md)) reads and writes the accumulator in memory on every call. The new form passes the accumulator and B by value, and the compiler emits the multiplication inline with the accumulator in registers from the start to the end of the loop over `k`.

## How it works, step by step

### 1. fp32: the micro-kernel

Each thread of the warp computes one column of C and accumulates 12 rows (`float acc[12]`). With 32 lanes:

- 12 accumulators x 2 zmm = **24 zmm** of accumulators;
- B: element `k` of the thread's column is contiguous across the lanes, so the row of B for the step is **2 zmm loads**;
- A: `a[k*12 + r]` does not depend on the lane; it is a scalar load with broadcast (`vbroadcastss`), 1 zmm at a time.

That is 24 + 2 + 1 = 27 zmm, no spills. Per step of `k`: 24 FMAs, 2 vector loads of B and 12 broadcasts of A. Each load of B is reused by 12 FMAs, and each broadcast of A, by 2.

The generated inner loop was checked in the assembly: 24 `vfmadd231ps`, 12 `vbroadcastss`, 2 masked loads, no spills.

### 2. fp32: packing and passes

`pack_a32` reorganizes A into micro-panels of 12 rows with `k` as the outermost index: the 12 values of A for a step are contiguous (48 bytes). `pack_b32` does the same for B in micro-panels of 32 columns: the row of B for a step is 128 contiguous bytes.

The loop over `k` is split into passes of `KC = 256`:

- a micro-panel of B has 256 x 32 x 4 = 32 KiB, the size of the L1D of Zen 4; it is re-read by the 8 micro-panels of A of the block (`MC = 96` rows);
- the block of A has 96 x 256 x 4 = 96 KiB, which stays in L2 (1 MiB per core) while the block traverses its `NC = 512` columns;
- neighboring blocks in the grid share the panel of B in L3.

`KC` of 64 and 128 and smaller values of `MC` and `NC` were tested and were worse (for example, `KC = 64`, `MC = 96`: between 1213 and 866 GFLOPS with all cores).

### 3. bf16: what came before

The first WMMA kernel written for the CPU packed A and B in contiguous 16x16 tiles, and each warp computed 4x4 accumulator fragments, with one library call per `mma_sync`. It reached **2.6 TFLOPS** (against 0.62 for `tensor_gemm`). The profile:

- 77% of the time in the native routine, 20% in the warp code;
- ~170 core cycles per `mma`, of which 131 in the routine; the ideal is 128 (128 `vdpbf16ps` at one per cycle).

Variations that did not help:

- **Other tile shapes per warp** (4x8, 8x4, 2x8 fragments): none better than 4x4.
- **Batched native routine**, 16 `mma` per call: in a microbenchmark, slower (63 against 34.5 ns per `mma`).
- **Inlining the pointer `mma` in the IR** (generating the `vdpbf16ps` in the warp code, with accumulators read and written in the per-thread arrays): worse, 1.9 TFLOPS. With the 16 `mma` of a step of `k` in the same basic block, LLVM interleaved the 16 (hundreds of accumulators live at the same time, with spills), and common subexpression elimination (CSE) began to share the broadcast of each pair of A among the `mma` that use the same fragment of A: the broadcast became a live value in a register, and not a memory operand folded into the instruction. Separating each `mma` with an empty `asm` with a side effect (which LLVM does not move or cross) led to 1834 to 1941 GFLOPS, without getting past the call. The change was reverted.

The lesson: the cost that remains in the call is loading and storing the 16 accumulators on every `mma`, and inlining does not solve it as long as the accumulator is memory. It has to be a value.

### 4. bf16: accumulators in registers

In the distribution of [25](25-tensor-cores.md), element `e` of a fragment, across the 32 lanes of a warp, is rows `2e` and `2e + 1` of the tile. So:

- element `e` of the accumulator, as a `<32 x float>` value, is 2 zmm of 16 lanes, each one a row of C: **the 8 elements are exactly the 16 accumulators of `vdpbf16ps`**;
- element `p` of B, as a `<32 x bfloat16>` value, is **one zmm** with rows `2p` and `2p + 1` of B, which a `vpermw` interleaves into the pairs of `k` that `vdpbf16ps` expects.

The new function `__cpusim_mmav_bf16_f32(j, a, b0..b7, c0..c7)` receives B and C by value and returns element `j` of D; `mma_sync` calls it 8 times. Since B and C are by-value arguments, their fragments do not have their address taken, and SROA promotes them to SSA values: the accumulator crosses the loop over `k` as a phi of vectors, in registers. A remains by pointer, and its fragment remains a per-thread array in memory (lane-major), from which the pairs are broadcast.

The `pure` attribute declares that the function does not write memory (it may read, like the A fragment): LLVM does not need to assume that a call alters memory, so loads before and after it can be combined, and a call whose result is not used can be eliminated. `convergent` prevents LLVM from making them dependent on more conditions, as for any warp operation.

In the SIMT pass, `emit_mmav`:

1. **Memoizes** by the operands (A, B, C) and by the basic block: the first of the 8 calls computes all of D, and the other 7 just take their element.
2. **Slices C** into rows of 16 lanes: `acc[r]` are lanes `(r % 2)*16` to `(r % 2)*16 + 15` of element `r / 2`.
3. For each pair of `k` (`p` from 0 to 7, the outer loop), applies **one `vpermw`** to element `p` of B and emits **16 `vdpbf16ps`**, one per row. With the pair as the outer loop, there are 16 independent accumulation chains (each `acc[r]` receives one instruction every 16) and only one permuted B live.
4. **Broadcasts the pairs of A from memory:** row `r` of A is in element `r / 2`, lanes `(r % 2)*16` onward, so pair `p` is at `(r/2)*W*2 + (r%2)*32 + 4p` bytes from the base of the warp. The 32-bit load is broadcast to the 16 lanes.
5. Joins the 16 rows back into the 8 elements of D (`concat_chunks`).

There is no call, nor load and store of the accumulators per `mma`: per `mma`, there are 8 `vpermw` and 128 `vdpbf16ps` with the broadcasts of A folded in.

### 5. bf16: the steps and the measured effect of each

The one-core measurements use `CPUSIM_CORES=1` (one core, its two SMT workers) with 256x256x4096; the peak of one core in bf16 is ~307 GFLOPS.

**(1) One accumulator per warp, A and C by value.** The kernel came to have a single accumulator fragment per warp: 16 zmm, which stay in registers for the whole loop (more accumulators would spill). The first version passed A by value too, and `emit_mmav` wrote A to a buffer on the stack to broadcast the pairs from there. GVN forwarded those stores to the loads (*store-to-load forwarding* in the IR): the broadcasts became `vpbroadcastd` from an xmm register, which are shuffles executed in the same floating-point units as the `vdpbf16ps`, and the values of A, B and the accumulators exceeded 32 zmm. Marking the stores as `volatile` did not prevent the instruction selection DAG from forwarding; marking the loads of the pairs as `volatile` eliminated the register broadcasts, but the loads remained separate instructions, not folded into `vdpbf16ps`. One core: 137 GFLOPS (before the volatile loads; the volatile version was only measured on all cores). All cores: 1807 to 1848 GFLOPS.

**(2) A by pointer, loaded one step ahead.** A came to be read from its own fragment, which `load_matrix_sync` filled. `load_matrix_sync` writes the fragment with 64-byte stores, and `mma` reads 4-byte pairs inside them; if the `mma` comes right after, each 4-byte load depends on a 64-byte store still in flight and has to wait for store-to-load forwarding from a wide store to a narrow load. The kernel came to load A one step of `k` ahead, alternating two fragments (`a0`, `a1`): when the `mma` reads a fragment, its stores have already retired. One core: 184 GFLOPS with A by pointer without the step ahead, 190 GFLOPS with it.

**(3) The order of the operands.** `vdpbf16ps zmm1, zmm2, zmm3/m512/m32bcst` only has the broadcast memory operand form (`{1to16}`) in the third source. With A as the second operand, LLVM could not fold the broadcast. Since each term `a0*b0 + a1*b1` is symmetric, and the product of two bfloat16 (8 significand bits each) is exact in `float` (24 bits), swapping A and B does not change the result. With A last, LLVM folded all the broadcasts: the JIT assembly has 260 `{1to16}` operands. One core: **287 GFLOPS, 93% of 307**.

**(4) All cores.** Best result between 3093 and 3173 GFLOPS (3172.8 in `samples/cpu_gemm`, 3137 in a long run): 73% to 97% of the peak measured right before, which varied from 3226 to 3936 GFLOPS with the temperature, and about 80% of the 3956 sustained. Cache blocking inside the warp (several tiles per warp, with passes of KB steps of `k` re-reading C) did not help: 2553, 2301 and 2024 GFLOPS for TI/TJ/KB = 4/4/16, 8/8/16 and 4/8/32. The remaining loss is not memory bandwidth.

The order of the runs matters: running the fp32 part first leaves the CPU hot, and the same bf16 run gave 1956 GFLOPS right after it and 3173 after a 20 s pause.

## The code

### fp32: parameters and packing

`samples/cpu_gemm.cu:39-57`

```cpp
constexpr int MR = 12, NR = 32;  // micro-tile: 12 rows x 32 columns, one column per thread
constexpr int KC = 256;          // k per pass
constexpr int MC = 96;           // rows per block (8 micro-panels of A)
constexpr int NC = 512;          // columns per block (16 micro-panels of B)

// Ap[(m / 12) * K * 12 + k * 12 + m % 12]: 12 rows of A, k-major.
__global__ void pack_a32(const float* A, float* Ap, int M, int K) {
    const long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i >= (long)M * K) return;
    const int m = int(i / K), k = int(i % K);
    Ap[(long)(m / MR) * K * MR + (long)k * MR + m % MR] = A[i];
}
// Bp[(n / 32) * K * 32 + k * 32 + n % 32]: 32 columns of B, k-major.
__global__ void pack_b32(const float* B, float* Bp, int K, int N) {
    const long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i >= (long)K * N) return;
    const int k = int(i / N), n = int(i % N);
    Bp[(long)(n / NR) * K * NR + (long)k * NR + n % NR] = B[i];
}
```

`MC = 96` is a multiple of `MR = 12`, and the sample uses `M = 4032` (a multiple of 96). The two packing kernels are plain CUDA, one thread per element.

### fp32: the kernel

`samples/cpu_gemm.cu:59-82`

```cpp
// One warp per block: C[m0 : m0 + MC, n0 : n0 + NC] = A * B.
__global__ void __launch_bounds__(32) sgemm(const float* Ap, const float* Bp, float* C, int N, int K) {
    const int lane = threadIdx.x;
    const int m0 = blockIdx.x * MC, n0 = blockIdx.y * NC;
    for (int pc = 0; pc < K; pc += KC)
        for (int jr = 0; jr < NC; jr += NR) {
            const float* b = Bp + (long)((n0 + jr) / NR) * K * NR + (long)pc * NR + lane;
            for (int ir = 0; ir < MC; ir += MR) {
                const float* a = Ap + (long)((m0 + ir) / MR) * K * MR + (long)pc * MR;
                float* c = C + (long)(m0 + ir) * N + n0 + jr + lane;
                float acc[MR];
#pragma unroll
                for (int r = 0; r < MR; ++r) acc[r] = pc == 0 ? 0.0f : c[(long)r * N];
#pragma unroll 4
                for (int k = 0; k < KC; ++k) {
                    const float bv = b[k * NR];
#pragma unroll
                    for (int r = 0; r < MR; ++r) acc[r] = fmaf(a[k * MR + r], bv, acc[r]);
                }
#pragma unroll
                for (int r = 0; r < MR; ++r) c[(long)r * N] = acc[r];
            }
        }
}
```

- `__launch_bounds__(32)` and the launch with 32 threads: a block is a warp, and the JIT specializes the kernel for that shape ([01](../iteration-1/01-specialized-jit.md)).
- `a` does not depend on `lane`: `a[k * MR + r]` is uniform, one broadcast per row. `b[k * NR]` has stride 1 across lanes: two contiguous zmm loads.
- `float acc[MR]` with all indices constant after the `#pragma unroll` becomes 12 `<32 x float>` values, and not a per-thread array in memory.
- `#pragma unroll 4` unrolls 4 steps of `k` per loop iteration.

### bf16: the declaration and `mma_sync`

`include/mma.h:30-38`

```cpp
// Element j of D for the calling lane, B and C by value, A by pointer. The
// compiler turns the 8 calls of one mma into one inline mma; the accumulator
// and B can stay in registers, and A's pairs are broadcast from its fragment
// in memory. On the CPU it pays to load A one step of k ahead of its mma.
#define __CPUSIM_MMAV_ARGS(T) T, T, T, T, T, T, T, T
__device__ __attribute__((convergent, pure)) float __cpusim_mmav_bf16_f32(int j, const __nv_bfloat16* a,
                                                                          __CPUSIM_MMAV_ARGS(unsigned short),
                                                                          __CPUSIM_MMAV_ARGS(float));
#undef __CPUSIM_MMAV_ARGS
```

`include/mma.h:152-165`

```cpp
template <typename LA, typename LB>
__device__ __forceinline__ void mma_sync(fragment<accumulator, 16, 16, 16, float>& d,
                                         const fragment<matrix_a, 16, 16, 16, __nv_bfloat16, LA>& a,
                                         const fragment<matrix_b, 16, 16, 16, __nv_bfloat16, LB>& b,
                                         const fragment<accumulator, 16, 16, 16, float>& c, bool satf = false) {
#define __CPUSIM_E(f, s) f.x[0] s, f.x[1] s, f.x[2] s, f.x[3] s, f.x[4] s, f.x[5] s, f.x[6] s, f.x[7] s
    float r[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) r[j] = __cpusim_mmav_bf16_f32(j, a.x, __CPUSIM_E(b, .__x), __CPUSIM_E(c, ));
#undef __CPUSIM_E
#pragma unroll
    for (int j = 0; j < 8; ++j) d.x[j] = r[j];
    if (satf) __cpusim::saturate(d);
}
```

- `__CPUSIM_E(b, .__x)` passes the 8 elements of B as `unsigned short` (the bits of the bfloat16); `__CPUSIM_E(c, )`, the 8 of C as `float`.
- `r[j]` receives the 8 results before any store to `d`: with `d` and `c` being the same fragment (`mma_sync(acc, a, b, acc)`), writing `d.x[0]` before the second call would change one of its operands, and the 8 calls would no longer be the same multiplication.

### bf16: the kernel

`samples/cpu_gemm.cu:101-120`

```cpp
__global__ void __launch_bounds__(32) bgemm(const __nv_bfloat16* Ap, const __nv_bfloat16* Bp, float* C, int N, int K) {
    const int tn = blockIdx.x, tm = blockIdx.y;  // consecutive blocks share the row panel of A
    const int KT = K / 16;
    const __nv_bfloat16* ap = Ap + (long)tm * KT * 256;
    const __nv_bfloat16* bp = Bp + (long)tn * KT * 256;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    wmma::fill_fragment(acc, 0.0f);
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a0, a1;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> b;
    wmma::load_matrix_sync(a0, ap, 16);
    for (int kt = 0; kt < KT; kt += 2) {  // K is a multiple of 32
        wmma::load_matrix_sync(a1, ap + (kt + 1) * 256, 16);
        wmma::load_matrix_sync(b, bp + kt * 256, 16);
        wmma::mma_sync(acc, a0, b, acc);
        if (kt + 2 < KT) wmma::load_matrix_sync(a0, ap + (kt + 2) * 256, 16);
        wmma::load_matrix_sync(b, bp + (kt + 1) * 256, 16);
        wmma::mma_sync(acc, a1, b, acc);
    }
    wmma::store_matrix_sync(C + (long)tm * 16 * N + tn * 16, acc, N, wmma::mem_row_major);
}
```

- One warp, one 16x16 tile of C, one accumulator fragment.
- `a0` and `a1` alternate: A for the next step is loaded before the `mma` of the current step (step 2 above). The loop is unrolled into two steps so that the two fragments have fixed roles, with no computed index.
- `blockIdx.x` traverses the columns of tiles: consecutive blocks share the row panel of A.

### `emit_mmav`

`src/compiler/simt.cpp:1756-1786`

```cpp
Value* WarpBuilder::emit_mmav(CallInst* call) {
    std::vector<Value*> key{b_.GetInsertBlock()};
    for (unsigned i = 1; i < call->arg_size(); ++i) key.push_back(call->getArgOperand(i));
    std::array<Value*, 8>& out = mmav_done_[key];
    const unsigned j = unsigned(cast<ConstantInt>(call->getArgOperand(0))->getZExtValue());
    if (out[0]) return out[j];

    auto* i8 = b_.getInt8Ty();
    auto* v32bf = FixedVectorType::get(b_.getBFloatTy(), 32);
    auto slice = [&](Value* v, unsigned first, unsigned n) {
        SmallVector<int, 32> idx;
        for (unsigned i = 0; i < n; ++i) idx.push_back(int(first + i));
        return b_.CreateShuffleVector(v, idx);
    };
    Value* pa = call->getArgOperand(1);
    auto* fa = dyn_cast<AllocaInst>(getUnderlyingObject(pa, 12));
    if (!fa || !private_elem_.count(fa)) return fail("tensor core fragment not in a per-thread array", call), nullptr;
    SplitAddress sa = split_address(pa);
    if (auto* lanes = dyn_cast<Constant>(sa.offsets)) {
        for (unsigned l = 0; l < W; ++l) {
            auto* e = dyn_cast_or_null<ConstantInt>(lanes->getAggregateElement(l));
            if (!e || e->getZExtValue() != l * 2) return fail("tensor core fragment chosen per lane", call), nullptr;
        }
    } else {
        return fail("tensor core fragment chosen per lane", call), nullptr;
    }
    Value *bv[8], *c[8];
    for (unsigned e = 0; e < 8; ++e) {
        bv[e] = vector(call->getArgOperand(2 + e));
        c[e] = vector(call->getArgOperand(10 + e));
    }
```

- The memoization key is the current basic block and arguments 1 to 17 (without `j`): the 8 calls of one `mma_sync` have the same operands and land on the same entry.
- The A fragment has to be a per-thread array with offset `l * 2` in lane `l`, that is, the same fragment in all lanes (the same requirement as `emit_mma`, [25](25-tensor-cores.md)).
- `vector(...)` gives the `W`-lane vector form of each per-thread value of B and C.

`src/compiler/simt.cpp:1787-1814`

```cpp
    SmallVector<uint16_t, 32> pairs;
    for (unsigned n = 0; n < 16; ++n) pairs.push_back(uint16_t(n)), pairs.push_back(uint16_t(16 + n));
    Constant* pair_idx = ConstantDataVector::get(ctx_, ArrayRef<uint16_t>(pairs));
    Function* dp = Intrinsic::getDeclaration(&m_, Intrinsic::x86_avx512bf16_dpbf16ps_512);
    Function* perm = Intrinsic::getDeclaration(&m_, Intrinsic::x86_avx512_permvar_hi_512);

    std::array<std::vector<Value*>, 8> parts;  // per element: 16-lane rows, in lane order
    for (unsigned w = 0; w < W / CPUSIM_WARP_SIZE; ++w) {
        const unsigned l0 = w * CPUSIM_WARP_SIZE;
        Value* abase = b_.CreateConstInBoundsGEP1_64(i8, sa.base, l0 * 2);
        Value* acc[16];
        for (unsigned r = 0; r < 16; ++r) acc[r] = slice(c[r / 2], l0 + (r % 2) * 16, 16);
        // k pair outermost: 16 independent accumulator chains, one B live.
        for (unsigned p = 0; p < 8; ++p) {
            Value* bp = b_.CreateBitCast(b_.CreateCall(perm, {slice(bv[p], l0, 32), pair_idx}), v32bf);
            for (unsigned r = 0; r < 16; ++r) {
                Value* at = b_.CreateConstInBoundsGEP1_64(i8, abase, (r / 2) * W * 2 + (r % 2) * 32 + 4 * p);
                Value* pr = b_.CreateAlignedLoad(b_.getInt32Ty(), at, Align(4));
                // A last: only the third operand has a {1to16} memory form
                // (each pair product is symmetric, so the order is free).
                acc[r] = b_.CreateCall(dp, {acc[r], bp, b_.CreateBitCast(b_.CreateVectorSplat(16, pr), v32bf)});
            }
        }
        for (unsigned e = 0; e < 8; ++e) parts[e].push_back(acc[2 * e]), parts[e].push_back(acc[2 * e + 1]);
    }
    for (unsigned e = 0; e < 8; ++e) out[e] = concat_chunks(parts[e]);
    return out[j];
}
```

- `pairs` is the `vpermw` index (`x86_avx512_permvar_hi_512`): position `2n` receives row `2p`, column `n`, and position `2n + 1`, row `2p + 1`, column `n`.
- `slice(c[r / 2], l0 + (r % 2) * 16, 16)`: the 16 lanes of the warp that are row `r` of C.
- The call to `x86_avx512bf16_dpbf16ps_512` has `(acc, B, A)`: A as the third operand, for the `{1to16}` form.
- The loop over `w` repeats everything for each warp of the vector; with the kernel launched with 32 threads, `W = 32` and there is one warp.

The dispatch sits next to the pointer path:

`src/compiler/simt.cpp:2489-2490`

```cpp
    if (is_mma(&inst)) return emit_mma(cast<CallInst>(&inst), mask);
    if (is_mmav(&inst)) return set(emit_mmav(cast<CallInst>(&inst)));
```

`set(...)` associates the result (element `j` of D, `<W x float>`) with the per-thread instruction, like any other value.

## When it applies (and when it does not)

- **The inline path** holds for every bfloat16 `mma_sync` with a `float` accumulator, in any kernel: `samples/tensor_gemm.cu` also came to use it without changing the source. Half, int8, uint8 and half accumulators continue with the call to `libcpusim_tensor` ([25](25-tensor-cores.md)).
- **The full gain** depends on the kernel cooperating: one accumulator per warp (16 zmm), A loaded one step ahead and one-warp blocks. With more live accumulator fragments per warp, they do not fit in 32 zmm and LLVM spills.
- `emit_mmav` does not use the mask: a warp with no active lanes also computes the `mma`, and the result is discarded by the masked stores. In an `mma_sync` inside an `if` that few warps take, this is extra work (`emit_mma` skips those warps).
- **The fp32 kernel** is a writing technique, not a compiler change: it holds for any kernel in which the size of the accumulators and the uniform operand are planned the same way.

## Results

### fp32 (`sgemm`, 4032 x 4096 x 4096)

| Configuration | GFLOPS | Efficiency |
|---|---|---|
| 1 core | 145 | 94% of 154 |
| All cores, best | 1608 to 1642 | ~83% of the ~1.96 TFLOPS peak measured at the time |
| All cores, hotter machine | ~1260 | 70% of the 1800 peak measured right before |

On the RTX 5060 Laptop, the same kernel compiled by `nvcc` gave 1540 GFLOPS, less than the 1642 on the CPU. The kernel was not written for the GPU (blocks of 32 threads, no shared memory), so the number says nothing about the peak of the GPU, only that the same source runs on both sides.

### bf16

| Version | 1 core | All cores |
|---|---|---|
| `tensor_gemm`, GPU style (2048^3) | | 0.62 TFLOPS |
| WMMA for the CPU, 4x4 fragments, library call | | 2.6 TFLOPS |
| Same kernel with the pointer `mma` inlined | | 1.9 TFLOPS (1834 to 1941 GFLOPS with the `asm` barriers) |
| (1) one accumulator, A and C by value | 137 GFLOPS | 1807 to 1848 GFLOPS |
| (2) A by pointer, one step ahead | 184 GFLOPS (without the step ahead), 190 GFLOPS | |
| (3) A as the third operand | 287 GFLOPS (93%) | 3093 to 3173 GFLOPS |

The all-cores numbers in the last row are 73% to 97% of the peak measured right before each run, and about 80% of the sustained peak of 3956 GFLOPS.

### Effect on `samples/tensor_gemm.cu`

Without changing the source, at 2048^3: bf16 from 0.62 to 1.06 TFLOPS. In the same run, fp16 gave 0.80 TFLOPS and int8 1.67 TOPS (they were 0.36 and 0.77 in [25](25-tensor-cores.md)); those types continue to use the library, and that difference is not explained by this change.

### Caveats on the measurements

- **Cycles per `mma`:** the ~170 core cycles were obtained by dividing thread cycles by 2 (two SMT workers per core). When only one of the two SMT siblings of a core is busy, this overestimates the efficiency.
- **Frequency:** sampling `scaling_cur_freq` in `/sys` did not give reliable values, and the system-wide `perf` counters are not allowed on this machine; for this reason the reference was always the peak measured right before.
- **Temperature:** on the laptop, results change by tens of percent between runs; the same bf16 run went from 1956 to 3173 GFLOPS depending on what ran before.

## Limitations and trade-offs

- **Only bfloat16 with a `float` accumulator** has the inline path. fp16, int8 and uint8 still go through one library call per `mma`, with the accumulator in memory.
- **2-byte fragment loads:** the `load_matrix_sync` of A and B becomes two masked 32-byte loads and one `vinserti64x4` per element; the two 16-lane pieces, contiguous in memory in the packed layout, are not joined into a 64-byte load.
- **One accumulator per warp:** the inline path is fast because the whole accumulator fits in registers; the kernel has to be written that way. More fragments per warp bring the spills back.
- **Efficiency with all cores:** depends on the temperature of the machine, and the remaining loss was not attributed; the cache blocking experiments indicate that it is not bandwidth.
- **The library's bfloat16 routine** still exists in `src/tensor/mma.cpp`, but `mma.h` no longer uses it.

## See also

- [00-basic-concepts.md](../00-basic-concepts.md): lanes, SMT, AVX-512.
- [01-specialized-jit.md](../iteration-1/01-specialized-jit.md) and [02-width-variants.md](../iteration-1/02-width-variants.md): the kernel specialized for blocks of 32 threads and the 32-lane variant.
- [05-address-decomposition.md](../iteration-1/05-address-decomposition.md): the lane-major layout of per-thread arrays, from which the pairs of A are broadcast.
- [25-tensor-cores.md](25-tensor-cores.md): the WMMA API, the distribution of the elements and the pointer path.
