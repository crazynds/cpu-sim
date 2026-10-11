// Tensor core operations, one benchmark each: C = A * B (n x n) through
// nvcuda::wmma for every input type and shape, or the same multiplication in
// plain C++ with OpenMP.
//
//   cuda: a WMMA kernel per operation: one warp per tile of C, walking k with
//         A loaded one step ahead. Built with nvcc it runs on the GPU, built
//         with cucc on the CPU simulator (AVX512-BF16/VNNI, see
//         docs/tutorials/01-setup.md for what each type needs).
//   omp:  for each type, a C++ function that computes the same product from
//         the same inputs with the same semantics (tf32 inputs truncated to a
//         10-bit mantissa, half accumulators rounded once at the end), with
//         OpenMP threads and loops the compiler vectorizes. The shape is a
//         tensor core detail, so omp runs each type once.
//
// Inputs are small integers, so every result is exact in every type: each
// line is checked against a reference on sampled rows.
//
//   build/cucc -O3 -std=c++20 -fopenmp -Xcompiler -march=native bench/tensor_ops.cu -o tensor_ops_cpu
//   nvcc -O3 -std=c++20 -arch=native -Xcompiler -fopenmp,-march=native bench/tensor_ops.cu -o tensor_ops_gpu
//   ./tensor_ops_cpu cuda|omp [n]        n: a multiple of 128 (default 2048)
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <type_traits>
#include <vector>

using namespace nvcuda;
namespace xp = nvcuda::wmma::experimental;

#define CHECK(call)                                                                            \
    do {                                                                                       \
        cudaError_t e_ = (call);                                                               \
        if (e_ != cudaSuccess) {                                                               \
            std::printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); \
            std::exit(1);                                                                      \
        }                                                                                      \
    } while (0)

static double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

/* ------------------------------------------------------------------ inputs */

// The logical values (small integers), the same for every type: A is n x n
// row-major, B is n x n row-major (av[k * n + j]).
struct Inputs {
    int n;
    std::vector<int> a, b;
};

static Inputs make_inputs(int n, int lo, int hi, uint32_t seed) {
    Inputs in{n, std::vector<int>(size_t(n) * n), std::vector<int>(size_t(n) * n)};
    uint32_t s = seed;
    auto next = [&] { return (s = s * 1664525u + 1013904223u) >> 8; };
    for (int& v : in.a) v = lo + int(next() % unsigned(hi - lo + 1));
    for (int& v : in.b) v = lo + int(next() % unsigned(hi - lo + 1));
    return in;
}

// kind: 0 multiply-add, 1 xor-popcount, 2 and-popcount (bits).
static int64_t reference(const Inputs& in, int i, int j, int kind) {
    int64_t s = 0;
    for (int k = 0; k < in.n; ++k) {
        const int x = in.a[size_t(i) * in.n + k], y = in.b[size_t(k) * in.n + j];
        s += kind == 1 ? (x ^ y) : kind == 2 ? (x & y) : int64_t(x) * y;
    }
    return s;
}

template <typename T>
static double as_double(T v) {
    if constexpr (std::is_same_v<T, __half>) return double(__half2float(v));
    else if constexpr (std::is_same_v<T, __nv_bfloat16>) return double(__bfloat162float(v));
    else return double(v);
}

// 16 sampled rows (the first and the last included), every column.
template <typename T>
static bool check(const Inputs& in, const std::vector<T>& c, int kind) {
    for (int t = 0; t < 16; ++t) {
        const int i = t == 0 ? 0 : t == 1 ? in.n - 1 : int((uint64_t(t) * 2654435761u) % unsigned(in.n));
        for (int j = 0; j < in.n; ++j)
            if (as_double(c[size_t(i) * in.n + j]) != double(reference(in, i, j, kind))) return false;
    }
    return true;
}

// Each value in the type the operation takes.
template <typename T>
static std::vector<T> convert(const std::vector<int>& v) {
    std::vector<T> out(v.size());
    for (size_t i = 0; i < v.size(); ++i) {
        if constexpr (std::is_same_v<T, __half>) out[i] = __float2half(float(v[i]));
        else if constexpr (std::is_same_v<T, __nv_bfloat16>) out[i] = __float2bfloat16(float(v[i]));
        else out[i] = T(v[i]);
    }
    return out;
}
// Sub-byte types: packed BITS bits per value, 32 / BITS values to a word;
// row-major (A) or column-major (B: transposed first).
static std::vector<unsigned> pack(const std::vector<int>& v, int n, int bits, bool transpose) {
    std::vector<unsigned> w(v.size() * bits / 32, 0);
    for (int r = 0; r < n; ++r)
        for (int c = 0; c < n; ++c) {
            const size_t i = size_t(r) * n + c;
            const int x = transpose ? v[size_t(c) * n + r] : v[i];
            w[i * bits / 32] |= (unsigned(x) & ((1u << bits) - 1)) << (i * bits % 32);
        }
    return w;
}

/* ------------------------------------------------------------------ cuda */

// One warp per TM x TN tile of C; A row-major, B row-major, C row-major.
template <typename TF, typename TS, typename TC, int TM, int TN, int TK>
__global__ void __launch_bounds__(32) wmma_gemm(const TS* a, const TS* b, TC* c, int n) {
    const int tm = blockIdx.y, tn = blockIdx.x;
    wmma::fragment<wmma::matrix_a, TM, TN, TK, TF, wmma::row_major> a0, a1;
    wmma::fragment<wmma::matrix_b, TM, TN, TK, TF, wmma::row_major> fb;
    wmma::fragment<wmma::accumulator, TM, TN, TK, TC> acc;
    wmma::fill_fragment(acc, TC(0));
    const TS* ap = a + size_t(tm) * TM * n;
    const TS* bp = b + size_t(tn) * TN;
    auto tf32 = [](auto& f) {
        if constexpr (std::is_same_v<TF, wmma::precision::tf32>)
            for (int i = 0; i < f.num_elements; ++i) f.x[i] = wmma::__float_to_tf32(f.x[i]);
    };
    wmma::load_matrix_sync(a0, ap, n);
    tf32(a0);
    for (int k = 0; k < n; k += 2 * TK) {  // A one step ahead
        wmma::load_matrix_sync(a1, ap + k + TK, n);
        tf32(a1);
        wmma::load_matrix_sync(fb, bp + size_t(k) * n, n);
        tf32(fb);
        wmma::mma_sync(acc, a0, fb, acc);
        if (k + 2 * TK < n) {
            wmma::load_matrix_sync(a0, ap + k + 2 * TK, n);
            tf32(a0);
        }
        wmma::load_matrix_sync(fb, bp + size_t(k + TK) * n, n);
        tf32(fb);
        wmma::mma_sync(acc, a1, fb, acc);
    }
    wmma::store_matrix_sync(c + size_t(tm) * TM * n + size_t(tn) * TN, acc, n, wmma::mem_row_major);
}

// Sub-byte types: A row-major, B column-major, packed; one warp per 8 x 8 tile.
// op: 0 mma_sync, 1 bmma xor, 2 bmma and.
template <typename T, int TK, int BITS>
__global__ void __launch_bounds__(32) subbyte_gemm(const unsigned* a, const unsigned* b, int* c, int n, int op) {
    const int tm = blockIdx.y, tn = blockIdx.x;
    wmma::fragment<wmma::matrix_a, 8, 8, TK, T, wmma::row_major> fa;
    wmma::fragment<wmma::matrix_b, 8, 8, TK, T, wmma::col_major> fb;
    wmma::fragment<wmma::accumulator, 8, 8, TK, int> acc;
    wmma::fill_fragment(acc, 0);
    for (int k = 0; k < n; k += TK) {
        wmma::load_matrix_sync(fa, a + (size_t(tm) * 8 * n + k) * BITS / 32, n);
        wmma::load_matrix_sync(fb, b + (size_t(tn) * 8 * n + k) * BITS / 32, n);
        if constexpr (BITS == 1)
            wmma::bmma_sync(acc, fa, fb, acc, op == 2 ? xp::bmmaBitOpAND : xp::bmmaBitOpXOR, xp::bmmaAccumulateOpPOPC);
        else
            wmma::mma_sync(acc, fa, fb, acc);
    }
    wmma::store_matrix_sync(c + size_t(tm) * 8 * n + size_t(tn) * 8, acc, n, wmma::mem_row_major);
}

struct Timing {
    double seconds;
    bool ok;
};

// Runs launch() once (warm-up: JIT, width choice), then 3 timed runs, best.
template <typename F>
static double time_device(F&& launch) {
    launch();
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());
    cudaEvent_t e0, e1;
    CHECK(cudaEventCreate(&e0));
    CHECK(cudaEventCreate(&e1));
    float best = 1e30f;
    for (int r = 0; r < 3; ++r) {
        CHECK(cudaEventRecord(e0));
        launch();
        CHECK(cudaEventRecord(e1));
        CHECK(cudaEventSynchronize(e1));
        float ms;
        CHECK(cudaEventElapsedTime(&ms, e0, e1));
        best = std::min(best, ms);
    }
    cudaEventDestroy(e0), cudaEventDestroy(e1);
    return best / 1e3;
}

template <typename TF, typename TS, typename TC, int TM, int TN, int TK>
static Timing cuda_gemm(const Inputs& in) {
    const int n = in.n;
    const std::vector<TS> a = convert<TS>(in.a), b = convert<TS>(in.b);
    TS *da, *db;
    TC* dc;
    CHECK(cudaMalloc(&da, a.size() * sizeof(TS)));
    CHECK(cudaMalloc(&db, b.size() * sizeof(TS)));
    CHECK(cudaMalloc(&dc, size_t(n) * n * sizeof(TC)));
    CHECK(cudaMemcpy(da, a.data(), a.size() * sizeof(TS), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(db, b.data(), b.size() * sizeof(TS), cudaMemcpyHostToDevice));
    const double s = time_device([&] { wmma_gemm<TF, TS, TC, TM, TN, TK><<<dim3(n / TN, n / TM), 32>>>(da, db, dc, n); });
    std::vector<TC> c(size_t(n) * n);
    CHECK(cudaMemcpy(c.data(), dc, c.size() * sizeof(TC), cudaMemcpyDeviceToHost));
    cudaFree(da), cudaFree(db), cudaFree(dc);
    return {s, check(in, c, 0)};
}

template <typename T, int TK, int BITS>
static Timing cuda_subbyte(const Inputs& in, int op) {
    const int n = in.n;
    const std::vector<unsigned> a = pack(in.a, n, BITS, false), b = pack(in.b, n, BITS, true);
    unsigned *da, *db;
    int* dc;
    CHECK(cudaMalloc(&da, a.size() * 4));
    CHECK(cudaMalloc(&db, b.size() * 4));
    CHECK(cudaMalloc(&dc, size_t(n) * n * 4));
    CHECK(cudaMemcpy(da, a.data(), a.size() * 4, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(db, b.data(), b.size() * 4, cudaMemcpyHostToDevice));
    const double s = time_device([&] { subbyte_gemm<T, TK, BITS><<<dim3(n / 8, n / 8), 32>>>(da, db, dc, n, op); });
    std::vector<int> c(size_t(n) * n);
    CHECK(cudaMemcpy(c.data(), dc, c.size() * 4, cudaMemcpyDeviceToHost));
    cudaFree(da), cudaFree(db), cudaFree(dc);
    return {s, check(in, c, op)};
}

/* ------------------------------------------------------------------ omp */

// C = A * B with each operand as the type gives it to the multiplier (T: fp32
// for the floating point types, int32 for the integer ones), accumulating in
// TA. Rows of C in parallel; per row, blocks of k and j so the block of B
// stays in cache; the innermost loop runs over j, contiguous, vectorized.
template <typename T, typename TA>
static void omp_gemm(const T* a, const T* b, TA* c, int n) {
    constexpr int KB = 256, JB = 512;
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        TA* ci = c + size_t(i) * n;
        for (int j = 0; j < n; ++j) ci[j] = TA(0);
        for (int k0 = 0; k0 < n; k0 += KB)
            for (int j0 = 0; j0 < n; j0 += JB)
                for (int k = k0; k < k0 + KB; ++k) {
                    const TA aik = TA(a[size_t(i) * n + k]);
                    const T* bk = b + size_t(k) * n;
#pragma omp simd
                    for (int j = j0; j < j0 + JB; ++j) ci[j] += aik * TA(bk[j]);
                }
    }
}

// Each omp run starts from the type's own storage (half, bf16, int8, packed
// 4-bit or bits): converting it to what the multiplier takes is part of the
// work, as loading fragments is for the tensor core.
template <typename TS, typename TC>
static Timing omp_float(const Inputs& in, bool tf32) {
    const int n = in.n;
    const std::vector<TS> a = convert<TS>(in.a), b = convert<TS>(in.b);
    std::vector<float> c(size_t(n) * n);
    std::vector<TC> out(c.size());
    auto run = [&] {
        std::vector<float> fa(a.size()), fb(b.size());
#pragma omp parallel for
        for (size_t i = 0; i < a.size(); ++i) {
            float x = as_double(a[i]), y = as_double(b[i]);
            if (tf32) {
                uint32_t u;
                std::memcpy(&u, &x, 4), u &= 0xffffe000u, std::memcpy(&x, &u, 4);
                std::memcpy(&u, &y, 4), u &= 0xffffe000u, std::memcpy(&y, &u, 4);
            }
            fa[i] = x, fb[i] = y;
        }
        omp_gemm(fa.data(), fb.data(), c.data(), n);
        if constexpr (std::is_same_v<TC, __half>) {
#pragma omp parallel for
            for (size_t i = 0; i < c.size(); ++i) out[i] = __float2half(c[i]);
        }
    };
    run();
    double best = 1e30;
    for (int r = 0; r < 3; ++r) {
        const double t0 = now_s();
        run();
        best = std::min(best, now_s() - t0);
    }
    if constexpr (std::is_same_v<TC, __half>) return {best, check(in, out, 0)};
    else return {best, check(in, c, 0)};
}

static Timing omp_double(const Inputs& in) {
    const int n = in.n;
    const std::vector<double> a = convert<double>(in.a), b = convert<double>(in.b);
    std::vector<double> c(size_t(n) * n);
    omp_gemm(a.data(), b.data(), c.data(), n);
    double best = 1e30;
    for (int r = 0; r < 3; ++r) {
        const double t0 = now_s();
        omp_gemm(a.data(), b.data(), c.data(), n);
        best = std::min(best, now_s() - t0);
    }
    return {best, check(in, c, 0)};
}

// int8, uint8 (bits 8) and the 4-bit types (unpacked first): int32 products.
template <typename T>
static Timing omp_int(const Inputs& in, int bits) {
    const int n = in.n;
    const std::vector<unsigned> pa = bits == 4 ? pack(in.a, n, 4, false) : std::vector<unsigned>();
    const std::vector<unsigned> pb = bits == 4 ? pack(in.b, n, 4, false) : std::vector<unsigned>();
    const std::vector<T> a8 = convert<T>(in.a), b8 = convert<T>(in.b);
    std::vector<int> c(size_t(n) * n);
    auto run = [&] {
        std::vector<int16_t> a(a8.size()), b(b8.size());
#pragma omp parallel for
        for (size_t i = 0; i < a.size(); ++i) {
            if (bits == 4) {  // unpack: 8 values to a word, sign-extended for s4
                const int sh = int(i % 8) * 4;
                int x = int((pa[i / 8] >> sh) & 15), y = int((pb[i / 8] >> sh) & 15);
                if (std::is_signed_v<T>) x = (x ^ 8) - 8, y = (y ^ 8) - 8;
                a[i] = int16_t(x), b[i] = int16_t(y);
            } else {
                a[i] = int16_t(a8[i]), b[i] = int16_t(b8[i]);
            }
        }
        omp_gemm(a.data(), b.data(), c.data(), n);
    };
    run();
    double best = 1e30;
    for (int r = 0; r < 3; ++r) {
        const double t0 = now_s();
        run();
        best = std::min(best, now_s() - t0);
    }
    return {best, check(in, c, 0)};
}

// Bits: c[i][j] = popcount(A row i op B column j), 64 bits at a time.
static Timing omp_bits(const Inputs& in, int op) {
    const int n = in.n, words = n / 64;
    std::vector<uint64_t> a(size_t(n) * words, 0), bt(size_t(n) * words, 0);  // B column-major
    for (int r = 0; r < n; ++r)
        for (int k = 0; k < n; ++k) {
            a[size_t(r) * words + k / 64] |= uint64_t(in.a[size_t(r) * n + k] & 1) << (k % 64);
            bt[size_t(r) * words + k / 64] |= uint64_t(in.b[size_t(k) * n + r] & 1) << (k % 64);
        }
    std::vector<int> c(size_t(n) * n);
    auto run = [&] {
#pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j) {
                const uint64_t* ai = a.data() + size_t(i) * words;
                const uint64_t* bj = bt.data() + size_t(j) * words;
                int s = 0;
#pragma omp simd reduction(+ : s)
                for (int w = 0; w < words; ++w) s += __builtin_popcountll(op == 2 ? ai[w] & bj[w] : ai[w] ^ bj[w]);
                c[size_t(i) * n + j] = s;
            }
    };
    run();
    double best = 1e30;
    for (int r = 0; r < 3; ++r) {
        const double t0 = now_s();
        run();
        best = std::min(best, now_s() - t0);
    }
    return {best, check(in, c, op)};
}

/* ------------------------------------------------------------------ driver */

static void report(const char* op, const char* shape, int n, const Timing& t) {
    const double ops = 2.0 * n * n * double(n);
    std::printf("%-22s %-10s %6d  %10.3f  %10.2f  %s\n", op, shape, n, t.seconds * 1e3, ops / t.seconds / 1e12,
                t.ok ? "ok" : "FAIL");
}

int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "cuda";
    const int n = argc > 2 ? std::atoi(argv[2]) : 2048;
    if (n % 128 != 0 || (mode != "cuda" && mode != "omp")) {
        std::printf("usage: tensor_ops cuda|omp [n, a multiple of 128]\n");
        return 1;
    }
    const Inputs small = make_inputs(n, -4, 4, 1), unit = make_inputs(n, -1, 1, 2);
    const Inputs s4 = make_inputs(n, -8, 7, 3), u4 = make_inputs(n, 0, 15, 4), bits = make_inputs(n, 0, 1, 5);
    const Inputs s8 = make_inputs(n, -128, 127, 6), u8 = make_inputs(n, 0, 255, 7);
    int cc = 0;
    if (mode == "cuda") {
        cudaDeviceProp p;
        CHECK(cudaGetDeviceProperties(&p, 0));
        cc = p.major * 10 + p.minor;
        std::printf("cuda on %s, compute capability %d.%d\n", p.name, p.major, p.minor);
    } else {
        std::printf("omp, %d threads\n", omp_get_max_threads());
    }
    std::printf("%-22s %-10s %6s  %10s  %10s  %s\n", "operation", "shape", "n", "time (ms)", "T(FL)OPS", "check");
    auto skip = [&](const char* op, int need) {
        if (cc >= need) return false;
        std::printf("%-22s skipped: needs compute capability %d.%d\n", op, need / 10, need % 10);
        return true;
    };
    if (mode == "cuda") {
        if (!skip("fp16 -> fp32", 70)) {
            report("fp16 -> fp32", "16x16x16", n, cuda_gemm<__half, __half, float, 16, 16, 16>(small));
            report("fp16 -> fp32", "32x8x16", n, cuda_gemm<__half, __half, float, 32, 8, 16>(small));
            report("fp16 -> fp32", "8x32x16", n, cuda_gemm<__half, __half, float, 8, 32, 16>(small));
            report("fp16 -> fp16", "16x16x16", n, cuda_gemm<__half, __half, __half, 16, 16, 16>(unit));
        }
        if (!skip("int8 -> int32", 72)) {
            report("int8 -> int32", "16x16x16", n, cuda_gemm<signed char, signed char, int, 16, 16, 16>(s8));
            report("int8 -> int32", "32x8x16", n, cuda_gemm<signed char, signed char, int, 32, 8, 16>(s8));
            report("int8 -> int32", "8x32x16", n, cuda_gemm<signed char, signed char, int, 8, 32, 16>(s8));
            report("uint8 -> int32", "16x16x16", n, cuda_gemm<unsigned char, unsigned char, int, 16, 16, 16>(u8));
        }
        if (!skip("int4 / bits", 75)) {
            report("int4 -> int32", "8x8x32", n, cuda_subbyte<xp::precision::s4, 32, 4>(s4, 0));
            report("uint4 -> int32", "8x8x32", n, cuda_subbyte<xp::precision::u4, 32, 4>(u4, 0));
            report("bits xor popc", "8x8x128", n, cuda_subbyte<xp::precision::b1, 128, 1>(bits, 1));
        }
        if (!skip("bf16 / tf32 / fp64", 80)) {
            report("bits and popc", "8x8x128", n, cuda_subbyte<xp::precision::b1, 128, 1>(bits, 2));
            report("bf16 -> fp32", "16x16x16", n, cuda_gemm<__nv_bfloat16, __nv_bfloat16, float, 16, 16, 16>(small));
            report("bf16 -> fp32", "32x8x16", n, cuda_gemm<__nv_bfloat16, __nv_bfloat16, float, 32, 8, 16>(small));
            report("bf16 -> fp32", "8x32x16", n, cuda_gemm<__nv_bfloat16, __nv_bfloat16, float, 8, 32, 16>(small));
            report("tf32 -> fp32", "16x16x8", n, cuda_gemm<wmma::precision::tf32, float, float, 16, 16, 8>(small));
            report("fp64 -> fp64", "8x8x4", n, cuda_gemm<double, double, double, 8, 8, 4>(small));
        }
    } else {
        report("fp16 -> fp32", "-", n, omp_float<__half, float>(small, false));
        report("fp16 -> fp16", "-", n, omp_float<__half, __half>(unit, false));
        report("int8 -> int32", "-", n, omp_int<signed char>(s8, 8));
        report("uint8 -> int32", "-", n, omp_int<unsigned char>(u8, 8));
        report("int4 -> int32", "-", n, omp_int<signed char>(s4, 4));
        report("uint4 -> int32", "-", n, omp_int<unsigned char>(u4, 4));
        report("bits xor popc", "-", n, omp_bits(bits, 1));
        report("bits and popc", "-", n, omp_bits(bits, 2));
        report("bf16 -> fp32", "-", n, omp_float<__nv_bfloat16, float>(small, false));
        report("tf32 -> fp32", "-", n, omp_float<float, float>(small, true));
        report("fp64 -> fp64", "-", n, omp_double(small));
    }
}
