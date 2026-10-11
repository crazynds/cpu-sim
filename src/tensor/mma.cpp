// Tensor core emulation: the 16x16x16 matrix multiply-accumulate of one warp
// (nvcuda::wmma::mma_sync, include/mma.h). The warp code the SIMT compiler
// generates calls these once per active warp, with the warp's fragments.
//
// Fragments live in the warp code's per-thread arrays, lane-major (element j
// of lane l at base[j * lanes + l], `lanes` the width of the variant). Every
// fragment of a 16x16 tile holds 8 elements per lane: element j of lane l is
// row 2j + l / 16, column l % 16. So element j of the warp's 32 lanes is rows
// 2j and 2j + 1, one after the other, and each row of A, B, C or D is 16
// contiguous values.
//
// The CPU's dot product instructions play the tensor core:
//   bf16: vdpbf16ps, 2 products per fp32 lane (AVX512-BF16);
//   int8: vpdpbusd, 4 products per int32 lane (AVX512-VNNI), unsigned x signed
//         bytes, the signedness fixed with a column or row sum;
//   fp16: no fp16 dot product on x86 without AVX512-FP16: vcvtph2ps and fma.
// Which ones the CPU has decides the compute capability the device reports,
// and the runtime refuses kernels that need more (common/topology.h), so a
// routine here only runs where its instructions exist.
#include <immintrin.h>

// Every loop below has a fixed count and is unrolled (#pragma GCC unroll): the
// accumulators and tiles are arrays of registers only if no index is left
// to compute at run time; otherwise they live on the stack.

#include <array>
#include <cstdint>
#include <cstring>

namespace {

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
// vpermb index: [B[4q][n], B[4q+1][n], B[4q+2][n], B[4q+3][n]] for each n,
// from rows 4q .. 4q+3 one after the other.
constexpr std::array<uint8_t, 64> kQuads = [] {
    std::array<uint8_t, 64> idx{};
    for (int n = 0; n < 16; ++n)
        for (int t = 0; t < 4; ++t) idx[4 * n + t] = uint8_t(16 * t + n);
    return idx;
}();

}  // namespace

extern "C" {

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

__attribute__((target("avx512f,f16c,fma"))) void __cpusim_tensor_f16_f32(float* d, const uint16_t* a, const uint16_t* b,
                                                                           const float* c, uint64_t lanes, uint32_t active) {
    if (!active) return;
    // A in float, read back as broadcast operands of the fmas.
    alignas(64) float af[256];
    #pragma GCC unroll 16
    for (int r = 0; r < 16; ++r)
        _mm512_store_ps(af + 16 * r, _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(row(a, lanes, r)))));
    __m512 acc[16];
    #pragma GCC unroll 16
    for (int r = 0; r < 16; ++r) acc[r] = _mm512_loadu_ps(row(c, lanes, r));
    // Half of B at a time: 16 accumulators + 8 rows of B in the 32 registers.
    #pragma GCC unroll 16
    for (int k0 = 0; k0 < 16; k0 += 8) {
        __m512 bk[8];
        #pragma GCC unroll 16
        for (int i = 0; i < 8; ++i)
            bk[i] = _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(row(b, lanes, k0 + i))));
        #pragma GCC unroll 16
        for (int r = 0; r < 16; ++r)
            #pragma GCC unroll 16
            for (int i = 0; i < 8; ++i) acc[r] = _mm512_fmadd_ps(_mm512_set1_ps(af[16 * r + k0 + i]), bk[i], acc[r]);
    }
    #pragma GCC unroll 16
    for (int r = 0; r < 16; ++r) _mm512_storeu_ps(row(d, lanes, r), acc[r]);
}

namespace {

// The 4 quads of B (4 rows of k each) in vpdpbusd's layout; `flip` xors every
// byte (0x80: signed <-> unsigned with a bias of 128).
__attribute__((target("avx512f,avx512bw,avx512vbmi"))) inline void b_quads(const uint8_t* b, uint64_t lanes, __m512i flip,
                                                                             __m512i* bq) {
    const __m512i quads = _mm512_loadu_si512(kQuads.data());
    #pragma GCC unroll 16
    for (int q = 0; q < 4; ++q) {
        __m256i lo = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + (2 * q) * lanes));
        __m256i hi = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + (2 * q + 1) * lanes));
        __m512i rows = _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1);
        bq[q] = _mm512_xor_si512(_mm512_permutexvar_epi8(quads, rows), flip);
    }
}

}  // namespace

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

// Unsigned x unsigned: B goes in as B - 128 (xor 0x80, signed), and
// 128 * (row sum of A) is added back.
__attribute__((target("avx512f,avx512bw,avx512vnni,avx512vbmi"))) void __cpusim_tensor_u8_s32(
    int32_t* d, const uint8_t* a, const uint8_t* b, const int32_t* c, uint64_t lanes, uint32_t active) {
    if (!active) return;
    __m512i bq[4];
    b_quads(b, lanes, _mm512_set1_epi8(char(0x80)), bq);
    __m512i acc[16];
    #pragma GCC unroll 16
    for (int r = 0; r < 16; ++r) {
        __m128i sums = _mm_sad_epu8(_mm_loadu_si128(reinterpret_cast<const __m128i*>(row(a, lanes, r))), _mm_setzero_si128());
        int32_t rowsum = int32_t(_mm_cvtsi128_si64(sums) + _mm_extract_epi64(sums, 1));
        acc[r] = _mm512_add_epi32(_mm512_loadu_si512(row(c, lanes, r)), _mm512_set1_epi32(rowsum << 7));
    }
    #pragma GCC unroll 16
    for (int r = 0; r < 16; ++r) {
        const uint8_t* ar = row(a, lanes, r);
        #pragma GCC unroll 16
        for (int q = 0; q < 4; ++q) {
            uint32_t quad;
            std::memcpy(&quad, ar + 4 * q, 4);
            acc[r] = _mm512_dpbusd_epi32(acc[r], _mm512_set1_epi32(int(quad)), bq[q]);
        }
    }
    #pragma GCC unroll 16
    for (int r = 0; r < 16; ++r) _mm512_storeu_si512(row(d, lanes, r), acc[r]);
}

}  // extern "C"
