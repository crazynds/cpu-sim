# 25. Tensor cores

> **In one sentence:** the `nvcuda::wmma` API (`mma.h`) now works in the 16x16x16 shape for half, bfloat16 and int8; the SIMT compiler replaces each `mma_sync` with a per-warp call to a native routine that uses the Zen 4 dot product instructions (`vdpbf16ps`, `vpdpbusd`, or F16C conversion plus FMA), and the device advertises the compute capability those instructions allow, refusing, like an older GPU, the kernels that ask for more. After this document, bfloat16 with a `float` accumulator came to be emitted inline in the warp code, without the call ([27](27-kernel-at-peak.md)).

The general concepts (lanes, variants, per-thread arrays) are in [00-basic-concepts.md](../00-basic-concepts.md). The 16-bit types are in [24](24-16-bit-types.md).

## The problem

On the GPU, a tensor core executes `D = A * B + C` on small matrices in a single warp instruction. CUDA exposes this through the WMMA API:

```cpp
wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a;
wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> b;
wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
wmma::load_matrix_sync(a, ptrA, lda);
wmma::load_matrix_sync(b, ptrB, ldb);
wmma::mma_sync(acc, a, b, acc);
```

A *fragment* is a 16x16 tile distributed among the 32 threads of the warp: each thread holds a part of the elements (8, for the types here), and `mma_sync` is a collective warp operation, in which each element of `D` depends on elements held by other threads. CUDA **does not specify** which element of the tile lives in which thread; the program can only access the elements through `load_matrix_sync`, `store_matrix_sync`, `fill_fragment` and element-wise operations (such as multiplying all of them by a scalar), which do not depend on the distribution.

Programs with `#include <mma.h>` did not compile. And a naive emulation, with each thread computing its elements through 16-term dot products reading the fragments of the other threads by shuffle, would not take advantage of the dot product instructions that Zen 4 has:

| Instruction | Extension | What it does per zmm register |
|---|---|---|
| `vdpbf16ps` | AVX512-BF16 | 16 `float` lanes: `acc += a0*b0 + a1*b1`, with pairs of bfloat16 |
| `vpdpbusd` | AVX512-VNNI | 16 `int32` lanes: `acc += sum of 4 products` of unsigned bytes by signed bytes |
| `vcvtph2ps` + `vfmadd231ps` | F16C, AVX-512 | there is no half dot product without AVX512-FP16: convert and use FMA |

## The idea

1. **Choose the distribution of the elements** so that, in the storage the compiler already uses for per-thread arrays, the fragment of a whole warp lies in contiguous rows of the matrix. A native routine then reads and writes the tile with simple vector loads.
2. **Treat `mma_sync` as a warp operation in the SIMT compiler**: instead of vectorizing the per-thread code, the compiler emits a call to the native routine for each warp present in the vector.
3. **Advertise the capability as a GPU does:** the compute capability of the device depends on which instructions the CPU has, and the runtime refuses a kernel whose type requires more, with the same error an old GPU would give.

## How it works, step by step

### 1. The distribution of the elements

For every fragment type (A, B and accumulator), element `j` (0 to 7) of lane `l` (0 to 31) is:

```text
row    = 2j + l / 16
column = l % 16
```

So element `j` of the 32 lanes covers two whole rows: lanes 0 to 15 have row `2j`, columns 0 to 15; lanes 16 to 31 have row `2j + 1`.

### 2. The lane-major storage

A per-thread array (such as `T x[8]` inside the fragment) becomes, in the vector code of `W` lanes, a single region with the copies of all the lanes interleaved by element ([05](../iteration-1/05-address-decomposition.md)): the byte offset `o` of lane `l` is at `base + o*W + l*elem`, where `elem` is the element size. For element `j`, `o = j*elem`, and the address is `base + (j*W + l)*elem`: the elements `j` of the `W` lanes are contiguous.

Combining this with the distribution: in warp `w` of the vector (lanes `32w` to `32w + 31`), element `j` occupies 32 contiguous positions starting at `base + (j*W + 32w)*elem`, and these 32 positions are rows `2j` and `2j + 1` of the tile, 16 values each. Example with `W = 64` (two warps) and a `float` fragment:

```text
offset in elements:  0 ..... 15 | 16 .... 31 | 32 .... 47 | 48 .... 63 | 64 ...
contents:            warp 0       warp 0       warp 1       warp 1       warp 0
                     row 0        row 1        row 0        row 1        row 2   (j = 1)
```

For warp `w`, row `r` is at `base_w + (r/2)*W + (r%2)*16`, with `base_w = base + 32w*elem`. The native routine receives `base_w` and `W` and reads each row with a 16-element load.

### 3. From `mma_sync` to the native routine

`mma.h` implements `mma_sync` by calling an external function marked `convergent`, with pointers to the four fragments:

```text
__cpusim_mma_f16_f32(d.x, a.x, b.x, c.x)
```

In the per-thread code, `d.x`, `a.x`... are pointers to per-thread arrays. The SIMT pass recognizes the call by the prefix `__cpusim_mma_` and emits, for each warp `w` of the vector:

```text
__cpusim_tensor_f16_f32(base_d + 32w*4, base_a + 32w*2, base_b + 32w*2, base_c + 32w*4, W, mask_bits[32w .. 32w+31])
```

The 32 mask bits of the warp go along: a warp with no active lane at that point (because the call is inside an `if` that only some warps take) does not touch its fragments. A warp with any active lane computes the whole tile, like the GPU, which requires the warp to be convergent at `mma_sync`.

For this, the offset of the fragment inside the per-thread array has to be **uniform**: the same in all lanes. `acc[i][j]` with constant or uniform `i` and `j` works; a fragment chosen by `threadIdx` (`acc[threadIdx.x % 2]`) has no single base per warp, and the compiler rejects the kernel with a message.

### 4. The native routines

They are in `libcpusim_tensor.so` (`src/tensor/mma.cpp`), linked by `cucc` and by the JIT with `--as-needed` (it only becomes a dependency of the kernel module if some kernel uses tensor cores).

**bfloat16 to float.** (A routine that `mma.h` no longer calls; the inline path of [27](27-kernel-at-peak.md) uses the same scheme of `vpermw` and broadcast of A.) `vdpbf16ps` multiplies pairs of bfloat16 and adds in `float`: for each 32-bit lane `n`, `acc[n] += a[2n]*b[2n] + a[2n+1]*b[2n+1]`. For the product `D[r][n] = sum_k A[r][k] * B[k][n]`, lane `n` of the accumulator of row `r` needs the pair `(B[2p][n], B[2p+1][n])` and the A operand needs the pair `(A[r][2p], A[r][2p+1])`, the same for all lanes:

- element `p` of fragment B is rows `2p` and `2p+1` (32 contiguous values, 64 bytes, one zmm load); a `vpermw` interleaves the two rows into pairs per column: `[B[2p][0], B[2p+1][0], B[2p][1], B[2p+1][1], ...]`;
- the pair of A is read as a 32-bit integer and broadcast to the 16 lanes (`vpbroadcastd`);
- 16 rows x 8 pairs = 128 `vdpbf16ps` per tile, with 16 accumulators and 8 registers of B (24 of the 32 zmm).

**half to float.** Without a half dot product, each row of A and of B is converted with `vcvtph2ps` (16 halfs from a ymm to 16 floats in a zmm). A is stored converted in a 1 KiB area on the stack and read as a broadcast operand of the FMA; B is processed in two halves of 8 rows, to fit with the 16 accumulators in the 32 registers. That is 256 `vfmadd231ps` per tile, and the products are exact in `float` (11 + 11 mantissa bits).

**int8 to int32.** `vpdpbusd` does, per 32-bit lane, `acc += sum of 4 products` of an **unsigned** byte of the first operand by a **signed** byte of the second. The four bytes of B of a lane have to be `B[4q..4q+3][n]`: two elements of the fragment (rows `4q` to `4q+3`) are loaded as 64 bytes and rearranged by `vpermb` (AVX512-VBMI). To correct the sign:

- **signed x signed:** A enters as `A xor 0x80`, which as an unsigned byte is `A + 128`. Then `(A + 128) . B = A . B + 128 * sum(B)`, and the accumulator starts at `C - 128 * (sum of each column of B)`. The column sum comes from another `vpdpbusd` with bytes of 1 in place of A.
- **unsigned x unsigned:** B enters as `B xor 0x80`, which as a signed byte is `B - 128`. Then `A . (B - 128) = A . B - 128 * sum(A)`, and the accumulator starts at `C + 128 * (sum of each row of A)`, computed with `vpsadbw` against zero (sum of absolute differences = sum of the bytes).

`vpdpbusd` does not saturate (the saturating form is `vpdpbusds`), so everything is arithmetic modulo 2^32 and the correction is exact, like the integer accumulation of the GPU without `satf`.

### 5. The compute capability

| What the CPU has | Level | Advertised compute capability |
|---|---|---|
| AVX512-BF16 and AVX512-VNNI (and VBMI) | bf16 | 12.0 |
| AVX512-VNNI (and VBMI), without BF16 | int8 | 7.5 |
| neither | fp16 | 7.0 |

On the GPU, the tensor core types appeared in different generations: half in sm_70, int8 in sm_72, bfloat16 in sm_80. `cpusim-devc` records in the manifest, per kernel, the lowest compute capability that the types used require (`sm=70`, `72` or `80`). At launch, the runtime compares it with that of the device and returns `cudaErrorNoKernelImageForDevice` if it is lower, exactly the error of a binary compiled for a newer GPU.

There is no fallback emulation (for example, bfloat16 with FMA on a CPU without AVX512-BF16): the decision was to refuse, as CUDA does, and to check only what CUDA checks. A well-written CUDA program queries `cudaGetDeviceProperties` and chooses the path; `samples/tensor_gemm.cu` does this.

Half is available at every level, even without a native dot product, because the compute capability is cumulative in CUDA: a 12.0 device that accepts bfloat16 has to accept half. Half runs with F16C and FMA, with lower throughput, and with exact products.

`CPUSIM_TENSOR=int8` or `fp16` lowers the level (never raises it above what the CPU has), to see how a program behaves on a smaller CPU. `cpu-smi` shows:

```text
Tensor Cores: fp16, int8, bf16 (AVX512-BF16 + AVX512-VNNI)
```

## The code

### The distribution in the header

`include/mma.h:83-85`

```cpp
// Row and column of element j of the calling lane.
static __device__ __forceinline__ unsigned row(int j) { return 2u * (unsigned)j + (__cpusim_laneid() >> 4); }
static __device__ __forceinline__ unsigned col() { return __cpusim_laneid() & 15u; }
```

`include/mma.h:95-100`

```cpp
// A and B: the layout is part of the fragment type. A is rows x k, B is k x columns.
template <typename Use, typename T>
__device__ __forceinline__ void load_matrix_sync(fragment<Use, 16, 16, 16, T, row_major>& f, const T* p, unsigned ldm) {
#pragma unroll
    for (int j = 0; j < 8; ++j) f.x[j] = p[__cpusim::row(j) * ldm + __cpusim::col()];
}
```

`load_matrix_sync` is ordinary per-thread code: each lane reads its 8 elements, and the SIMT compiler vectorizes it like any other load ([12](../iteration-2/12-static-access-analysis.md)). For element `j`, lanes 0 to 15 read 16 consecutive values of a memory row and lanes 16 to 31 read those of the next row. The `col_major` layout only swaps `row` and `col` in the read; the tile in the fragment is the same.

`include/mma.h:144-151`

```cpp
template <typename LA, typename LB>
__device__ __forceinline__ void mma_sync(fragment<accumulator, 16, 16, 16, float>& d,
                                         const fragment<matrix_a, 16, 16, 16, __half, LA>& a,
                                         const fragment<matrix_b, 16, 16, 16, __half, LB>& b,
                                         const fragment<accumulator, 16, 16, 16, float>& c, bool satf = false) {
    __cpusim_mma_f16_f32(d.x, a.x, b.x, c.x);
    if (satf) __cpusim::saturate(d);
}
```

This is the half `mma_sync` with a `float` accumulator; those for int8 and uint8 (`__cpusim_mma_s8_s32`, `__cpusim_mma_u8_s32`) and the one for half accumulators (which converts to a temporary `float` fragment and calls `__cpusim_mma_f16_f32`) have the same form: one call with pointers to the four fragments.

bfloat16 with a `float` accumulator no longer follows this path. It uses another function, `__cpusim_mmav_bf16_f32`, which receives B and C by value and A by pointer, and is called 8 times per `mma_sync`, once per element of D:

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

The compiler merges the 8 calls into a single multiplication emitted inline in the warp code (`emit_mmav`), without calling the library, with the accumulator and B in registers. That path, the reasons for each choice and the measured gain are in [27](27-kernel-at-peak.md). The rest of this document describes the pointer path, which remains the one for half, int8 and uint8; the library's native bfloat16 routine (below) still exists, but `mma.h` no longer uses it.

### The recognition in the SIMT pass

`src/compiler/simt.h:40-57`

```cpp
// Tensor cores: `__cpusim_mma_<types>(d, a, b, c)`, pointers to the fragments
// (per-thread arrays) of one warp's 16x16x16 multiply-accumulate (mma.h). The
// warp code calls `__cpusim_tensor_<types>` of libcpusim_tensor per warp.
constexpr const char* kMmaPrefix = "__cpusim_mma_";
constexpr const char* kTensorPrefix = "__cpusim_tensor_";
// `float __cpusim_mmav_<types>(int j, a, b0..b7, c0..c7)`: element j of
// D = A * B + C for the calling thread, A by pointer (its fragment), B and C
// by value; the 8 calls of one mma_sync share their operands and become one
// inline mma (emit_mmav). Fragments passed by value can live in registers.
constexpr const char* kMmavPrefix = "__cpusim_mmav_";
inline bool is_mmav(const llvm::Value* v) {
    auto* call = llvm::dyn_cast<llvm::CallInst>(v);
    return call && call->getCalledFunction() && call->getCalledFunction()->getName().starts_with(kMmavPrefix);
}
inline bool is_mma(const llvm::Value* v) {
    auto* call = llvm::dyn_cast<llvm::CallInst>(v);
    return call && call->getCalledFunction() && call->getCalledFunction()->getName().starts_with(kMmaPrefix);
}
```

Two places in the SIMT pass came to accept these calls (the pointer ones, `is_mma`, and the by-value ones, `is_mmav`). `check_supported`, which rejects calls to functions that could not be inlined, lets both through (`src/compiler/simt.cpp:441`). And `check_private_array`, which only accepts a per-thread array if all uses are element loads and stores (a requirement of the lane-major layout), also accepts its use as an argument of them (the A fragment of `__cpusim_mmav_bf16_f32` remains a per-thread array passed by pointer):

`src/compiler/simt.cpp:1163-1167`

```cpp
            } else if (auto* ii = dyn_cast<IntrinsicInst>(u); ii && ii->isLifetimeStartOrEnd()) {
            } else if (is_mma(u) || is_mmav(u)) {  // a tensor core fragment, handed over whole (emit_mma)
            } else {
                return fail("per-thread array used other than by element loads and stores", u), false;
            }
```

### The emission of the calls

`src/compiler/simt.cpp:1704-1733`

```cpp
// __cpusim_mma_<types>(d, a, b, c): one warp's 16x16x16 multiply-accumulate
// on fragments in per-thread arrays. Lane-major, element j of the warp's 32
// lanes is contiguous, so each warp of the vector hands its part of the four
// arrays to the native routine (libcpusim_tensor), with the vector width and
// its 32 bits of the mask: a warp with no lane here leaves its fragments alone.
bool WarpBuilder::emit_mma(CallInst* call, Value* mask) {
    auto* ptr = PointerType::get(ctx_, 0);
    auto* i32 = b_.getInt32Ty();
    auto* i64 = b_.getInt64Ty();
    const std::string native =
        kTensorPrefix + call->getCalledFunction()->getName().drop_front(StringRef(kMmaPrefix).size()).str();
    FunctionCallee fn = m_.getOrInsertFunction(native, FunctionType::get(b_.getVoidTy(), {ptr, ptr, ptr, ptr, i64, i32}, false));
    Value* base[4];
    uint64_t elem[4];
    for (unsigned i = 0; i < 4; ++i) {
        Value* p = call->getArgOperand(i);
        auto* a = dyn_cast<AllocaInst>(getUnderlyingObject(p, 12));
        if (!a || !private_elem_.count(a)) return fail("tensor core fragment not in a per-thread array", call), false;
        SplitAddress s = split_address(p);
        // The same fragment in every lane: each lane's copy at l * elem.
        elem[i] = private_elem_[a];
        auto* lanes = dyn_cast<Constant>(s.offsets);
        for (unsigned l = 0; lanes && l < W; ++l) {
            auto* e = dyn_cast_or_null<ConstantInt>(lanes->getAggregateElement(l));
            if (!e || e->getZExtValue() != l * elem[i]) lanes = nullptr;
        }
        if (!lanes) return fail("tensor core fragment chosen per lane (an array of fragments indexed by thread)", call), false;
        base[i] = s.base;
    }
    Value* bits = b_.CreateBitCast(mask, b_.getIntNTy(W));
```

- The name of the routine comes from the name of the call: `__cpusim_mma_f16_f32` becomes `__cpusim_tensor_f16_f32`.
- `split_address` ([05](../iteration-1/05-address-decomposition.md)) separates the address into a uniform base plus a per-lane offset. In a per-thread array, the offset of lane `l` is `l*elem` plus the offset of the element inside the array; the loop requires it to be exactly `l*elem`, that is, that the fragment starts at the same point of the array in all lanes. The base `s.base` already includes the uniform offset of the fragment (for example, `acc[1][0]` inside `acc[2][2]`).
- The `W`-bit mask becomes an integer (`bits`).

`src/compiler/simt.cpp:1734-1743`

```cpp
    for (unsigned w = 0; w < W / CPUSIM_WARP_SIZE; ++w) {
        SmallVector<Value*, 6> args;
        for (unsigned i = 0; i < 4; ++i)
            args.push_back(b_.CreateConstInBoundsGEP1_64(b_.getInt8Ty(), base[i], w * CPUSIM_WARP_SIZE * elem[i]));
        args.push_back(ConstantInt::get(i64, W));
        args.push_back(b_.CreateTrunc(b_.CreateLShr(bits, uint64_t(w * CPUSIM_WARP_SIZE)), i32));
        b_.CreateCall(fn, args);
    }
    return true;
}
```

- Warp `w` receives the base of each fragment offset by `32w` elements and mask bits `32w` to `32w + 31`.
- `W` goes as an argument because the same routine serves all widths.

Right after `emit_mma` (`src/compiler/simt.cpp:1745-1814`) comes `emit_mmav`, which handles the by-value bfloat16 calls by emitting the multiplication inline in the warp code, instead of calling the library; it is described in [27](27-kernel-at-peak.md).

### The bfloat16 routine

`src/tensor/mma.cpp:32-42`

```cpp
template <typename T>
inline T* row(T* base, uint64_t lanes, int r) {
    return base + (r >> 1) * lanes + (r & 1) * 16;
}

// vpermw index: [B[2p][0], B[2p+1][0], B[2p][1], ...] from rows 2p, 2p+1.
constexpr std::array<uint16_t, 32> kPairs = [] {
    std::array<uint16_t, 32> idx{};
    for (int n = 0; n < 16; ++n) idx[2 * n] = uint16_t(n), idx[2 * n + 1] = uint16_t(16 + n);
    return idx;
}();
```

`row` is the formula from step 2. `kPairs` is the `vpermw` index: position `2n` receives value `n` (row `2p`, column `n`) and position `2n+1` receives `16 + n` (row `2p+1`, column `n`).

`src/tensor/mma.cpp:56-79`

```cpp
__attribute__((target("avx512f,avx512bw,avx512bf16"))) void __cpusim_tensor_bf16_f32(float* d, const uint16_t* a,
                                                                                       const uint16_t* b, const float* c,
                                                                                       uint64_t lanes, uint32_t active) {
    if (!active) return;
    const __m512i pairs = _mm512_loadu_si512(kPairs.data());
    __m512bh bp[8];
    #pragma GCC unroll 16
    for (int p = 0; p < 8; ++p) bp[p] = (__m512bh)_mm512_permutexvar_epi16(pairs, _mm512_loadu_si512(b + p * lanes));
    __m512 acc[16];
    #pragma GCC unroll 16
    for (int r = 0; r < 16; ++r) acc[r] = _mm512_loadu_ps(row(c, lanes, r));
    #pragma GCC unroll 16
    for (int r = 0; r < 16; ++r) {
        const uint16_t* ar = row(a, lanes, r);
        #pragma GCC unroll 16
        for (int p = 0; p < 8; ++p) {
            uint32_t pair;
            std::memcpy(&pair, ar + 2 * p, 4);
            acc[r] = _mm512_dpbf16_ps(acc[r], (__m512bh)_mm512_set1_epi32(int(pair)), bp[p]);
        }
    }
    #pragma GCC unroll 16
    for (int r = 0; r < 16; ++r) _mm512_storeu_ps(row(d, lanes, r), acc[r]);
}
```

The `target(...)` allows compiling the library for any x86-64 and using the extensions only in this function; it is only called when the CPU has them, because the runtime previously refused the kernels that would need them. The `#pragma GCC unroll` are mandatory: `acc[16]` and `bp[8]` only become registers if every index is constant after unrolling; with an index computed at run time, GCC keeps them on the stack (see Results).

### The sign correction in int8

`src/tensor/mma.cpp:126-152`

```cpp
// Signed x signed: vpdpbusd multiplies unsigned by signed bytes, so A goes in
// as A + 128 (xor 0x80) and the extra 128 * (column sum of B) comes off first.
__attribute__((target("avx512f,avx512bw,avx512vnni,avx512vbmi"))) void __cpusim_tensor_s8_s32(
    int32_t* d, const int8_t* a, const int8_t* b, const int32_t* c, uint64_t lanes, uint32_t active) {
    if (!active) return;
    __m512i bq[4];
    b_quads(reinterpret_cast<const uint8_t*>(b), lanes, _mm512_setzero_si512(), bq);
    __m512i colsum = _mm512_setzero_si512();
    #pragma GCC unroll 16
    for (int q = 0; q < 4; ++q) colsum = _mm512_dpbusd_epi32(colsum, _mm512_set1_epi8(1), bq[q]);
    const __m512i bias = _mm512_slli_epi32(colsum, 7);
    __m512i acc[16];
    #pragma GCC unroll 16
    for (int r = 0; r < 16; ++r) acc[r] = _mm512_sub_epi32(_mm512_loadu_si512(row(c, lanes, r)), bias);
    #pragma GCC unroll 16
    for (int r = 0; r < 16; ++r) {
        const int8_t* ar = row(a, lanes, r);
        #pragma GCC unroll 16
        for (int q = 0; q < 4; ++q) {
            uint32_t quad;
            std::memcpy(&quad, ar + 4 * q, 4);
            acc[r] = _mm512_dpbusd_epi32(acc[r], _mm512_set1_epi32(int(quad ^ 0x80808080u)), bq[q]);
        }
    }
    #pragma GCC unroll 16
    for (int r = 0; r < 16; ++r) _mm512_storeu_si512(row(d, lanes, r), acc[r]);
}
```

- `colsum`: `vpdpbusd` with the unsigned operand all 1s adds the 4 signed bytes of each lane; summing the 4 quads, each lane `n` has the sum of column `n` of B (16 values).
- `bias = colsum << 7` is `128 * colsum`, subtracted from the initial accumulator.
- `quad ^ 0x80808080u` applies the `xor 0x80` to the 4 bytes of A at once before broadcasting them.

### The compute capability

`src/common/topology.cpp:89-105`

```cpp
TensorLevel tensor_level() {
    __builtin_cpu_init();
    TensorLevel level = TensorLevel::Fp16;
    // vpdpbusd and the byte permutes (vpermb) of the int8 path.
    if (__builtin_cpu_supports("avx512vnni") && __builtin_cpu_supports("avx512vbmi")) {
        level = TensorLevel::Int8;
        if (__builtin_cpu_supports("avx512bf16")) level = TensorLevel::Bf16;
    }
    if (const char* v = std::getenv("CPUSIM_TENSOR")) {
        TensorLevel want = level;
        if (std::strcmp(v, "fp16") == 0) want = TensorLevel::Fp16;
        else if (std::strcmp(v, "int8") == 0) want = TensorLevel::Int8;
        else if (std::strcmp(v, "bf16") == 0) want = TensorLevel::Bf16;
        if (want < level) level = want;
    }
    return level;
}
```

`src/common/topology.cpp:115-123`

```cpp
int compute_capability(TensorLevel level) {
    // The level a GPU first had these tensor core types at; with everything,
    // the architecture of the GPU this machine pairs with (RTX 50, sm_120).
    switch (level) {
        case TensorLevel::Bf16: return 120;
        case TensorLevel::Int8: return 75;
        default: return 70;
    }
}
```

The bf16 level only exists inside int8 (a CPU with AVX512-BF16 and without VNNI stays at fp16), because each level includes those below it. With everything available, the device advertises 12.0, the same compute capability as the RTX 5060 used as a reference.

`src/compiler/main.cpp:413-421`

```cpp
        // The compute capability the kernel's tensor core types need, as on
        // a GPU: fp16 since sm_70, int8 since sm_72, bf16 since sm_80.
        int min_sm = 0;
        for (BasicBlock& bb : *k.scalar)
            for (Instruction& inst : bb)
                if (cpusim::is_mma(&inst) || cpusim::is_mmav(&inst)) {
                    StringRef types = cast<CallInst>(inst).getCalledFunction()->getName();
                    min_sm = std::max(min_sm, types.contains("bf16") ? 80 : types.contains("8_") ? 72 : 70);
                }
```

`"8_"` matches `s8_s32` and `u8_s32`; `__cpusim_mmav_bf16_f32` contains `bf16` and requires `sm=80`, like the pointer call it replaced. The value goes into the kernel's line in the manifest as `sm=NN`.

`src/cudart/runtime.cpp:958-961`

```cpp
    // Tensor core types the CPU cannot run natively (no AVX512-BF16 for bf16,
    // no AVX512-VNNI for int8): like a binary built for a newer GPU.
    if (f.kernel->min_sm > d->properties().major * 10 + d->properties().minor)
        return record(Error::NoKernelImageForDevice);
```

## When it applies (and when it does not)

It applies to `nvcuda::wmma` in the 16x16x16 shape with:

- half x half, `float` or half accumulator (the half accumulator is converted to `float`, computed and rounded once);
- bfloat16 x bfloat16, `float` accumulator;
- int8 x int8 and uint8 x uint8, `int` accumulator.

Not implemented: the m32n8k16 and m8n32k16 shapes, tf32, `double`, the experimental types of fewer than 8 bits and `satf` for integer accumulators.

The compiler rejects a kernel in which the fragment used in an `mma_sync` depends on the lane (an array of fragments indexed by thread). Arrays of fragments indexed by constants or uniform values work.

## Results

### Test

`tests/cuda/wmma.cu` covers the 5 type combinations, `row_major` and `col_major` layouts for A and B, `ldm` with padding, C loaded and D stored in both layouts, tiles going through shared memory with barriers, arrays of fragments, `alpha * AB + beta * C` with element-wise operations, and a final step in which only the even warps multiply (the fragments of the odd ones have to remain intact). The inputs are small integers, so all sums are exact and the result is compared without tolerance. It runs in all modes and two more times, with `CPUSIM_TENSOR=int8` and `CPUSIM_TENSOR=fp16`, in which the types above the level have to fail with `cudaErrorNoKernelImageForDevice`:

`CMakeLists.txt:138-140`

```cmake
    # Tensor cores on CPUs with fewer dot product instructions: lower compute
    # capability, and the kernels that need more must not launch.
    set_tests_properties(cuda_wmma PROPERTIES ENVIRONMENT "FEATURE_ENVS=CPUSIM_TENSOR=int8 CPUSIM_TENSOR=fp16")
```

### Performance: `samples/tensor_gemm.cu`

The sample multiplies square matrices with blocks of 4 warps, each block computing a 64x64 tile of C and each warp 2x2 accumulator fragments; A and B go through shared memory. The same source compiles with `nvcc`. The story, at 2048^3 with bfloat16:

1. **First version: 0.19 TFLOPS.** `perf` showed the native routine with only 7.45% of the time, and the top was scalar `idiv`: 640 `idiv` instructions in the hot function. It was the division by a constant in divergent code, fixed in the compiler ([26](26-compiler-and-runtime-changes.md), part a).
2. **Scattered 2-byte stores.** The copy was the usual block-stride loop, `for (e = threadIdx.x; e < N; e += blockDim.x)`. The counter `e` is a variable value (each lane has its own) that evolves through the loop, and the access analysis does not prove that the address `As[e / 16][e % 16]` is contiguous across lanes; the store becomes a scatter. AVX-512 only has scatter for 32- and 64-bit elements, so a bfloat16 scatter is scalarized into one `vpextrw` to memory per lane. Rewriting the copy with a uniform counter, `e = i*128 + threadIdx.x`, the address becomes a uniform base plus `threadIdx.x`, which the analysis recognizes, and with B stored in sub-tiles of 16 columns: **0.49 TFLOPS** (fp16 0.45 TFLOPS, int8 0.66 TOPS).
3. **Native routines without unrolling.** GCC did not unroll the loops of the routines: the accumulators stayed on the stack, and the binary had only 8 `vdpbf16ps` (the inner loop unrolled, the outer not). With `#pragma GCC unroll` on all loops, the binary has 128 `vdpbf16ps` (the 16 x 8 of the tile). In a single-thread microbenchmark, per 16x16x16 `mma`:

| Type | Before | After |
|---|---|---|
| bf16 | 62.9 ns | 31.1 ns |
| fp16 | 116.1 ns | 86.0 ns |
| int8 | 34.2 ns | 24.5 ns |

Final result, compared with the RTX 5060 Laptop running the same source compiled by `nvcc`:

| Size | Type | cucc (CPU) | RTX 5060 |
|---|---|---|---|
| 2048^3 | bf16 | 0.62 TFLOPS | 15.34 TFLOPS |
| 2048^3 | fp16 | 0.36 TFLOPS | 17.63 TFLOPS |
| 2048^3 | int8 | 0.77 TOPS | 16.46 TOPS |
| 4096^3 | bf16 | 0.55 TFLOPS | 14.06 TFLOPS |
| 4096^3 | fp16 | 0.36 TFLOPS | 15.74 TFLOPS |
| 4096^3 | int8 | 0.59 TOPS | 16.26 TOPS |

These are the numbers from the end of this work, kept as a record. With the inline bfloat16 path ([27](27-kernel-at-peak.md)), bf16 at 2048^3 went to 1.06 TFLOPS.

The GPU is between 21 and 49 times ahead, depending on the type and size. fp16 is the slowest on the CPU, as expected: without a native dot product, each tile is 256 FMAs plus the conversions, against 128 `vdpbf16ps`.

## Limitations and trade-offs

- **No fallback emulation:** on a CPU without AVX512-BF16, a bfloat16 kernel does not run, even though it could be emulated with FMA. This is the behavior of a GPU without support for the type, and the program has to handle the error or query the compute capability.
- **bfloat16 rounding:** the order of the sums inside `vdpbf16ps` and the instruction's handling of subnormals are not the same as those of a GPU tensor core (which are also not specified by CUDA). With general inputs, the `float` result can differ from the GPU's in the last bits; the test uses small integers so that the comparison is exact.
- **Single shape:** only 16x16x16.
- **Compile time:** a kernel with WMMA generates large wide variants; see the generic variant budget in [26](26-compiler-and-runtime-changes.md).

## See also

- [00-basic-concepts.md](../00-basic-concepts.md): lanes, masks, AVX-512.
- [05-address-decomposition.md](../iteration-1/05-address-decomposition.md): `split_address` and the lane-major layout of per-thread arrays.
- [12-static-access-analysis.md](../iteration-2/12-static-access-analysis.md): why the uniform counter in the copy avoids the scatter.
- [24-16-bit-types.md](24-16-bit-types.md): `__half` and `__nv_bfloat16`.
- [26-compiler-and-runtime-changes.md](26-compiler-and-runtime-changes.md): the division by a constant, the variant budget and the JIT cache, found in this work.
- [27-kernel-at-peak.md](27-kernel-at-peak.md): the inline bfloat16 `mma_sync`, with accumulators in registers, and a GEMM written for the CPU.
